#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp


def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


def segment_softmax(src: torch.Tensor, index: torch.Tensor, num_segments: int, eps: float = 1e-12):
    if src.numel() == 0:
        return src

    # max per segment (for stability)
    max_per = torch.full((num_segments,), -float("inf"), device=src.device, dtype=src.dtype)

    # torch >= 2.0 supports scatter_reduce_
    if hasattr(max_per, "scatter_reduce_"):
        max_per.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    else:
        raise RuntimeError("Your PyTorch is too old (no scatter_reduce_). Please upgrade to torch>=2.0.")

    src_shift = src - max_per[index]
    exp = torch.exp(src_shift)

    denom = torch.zeros((num_segments,), device=src.device, dtype=src.dtype)
    denom.scatter_add_(0, index, exp)

    return exp / (denom[index] + eps)


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0, alpha=0.2, concat=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.W, gain=1.414)

        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a, gain=1.414)

        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @torch.no_grad()
    def _csr_to_coo_row(self, indptr: torch.Tensor, N: int):
        # row index for each edge: [E]
        deg = indptr[1:] - indptr[:-1]  # [N]
        row = torch.repeat_interleave(torch.arange(N, device=indptr.device), deg)
        return row

    def forward(self, input_h: torch.Tensor, indptr: torch.Tensor, indices: torch.Tensor):
        assert indptr.dim() == 1 and indices.dim() == 1
        N = input_h.size(0)
        assert indptr.numel() == N + 1

        # (可选) 输入特征 dropout
        x = F.dropout(input_h, p=self.dropout, training=self.training)

        # linear
        h = x @ self.W  # [N, Fout]
        if self.bias is not None:
            h = h + self.bias

        E = indices.numel()
        if E == 0:
            # 没有边：输出全 0（如果你想保留原特征，可在外面加 residual）
            return torch.zeros((N, self.out_features), device=h.device, dtype=h.dtype)

        # CSR -> COO rows
        row = self._csr_to_coo_row(indptr, N)  # [E]
        col = indices  # [E]

        # attention logits on edges: e_ij = LeakyReLU( a^T [h_i || h_j] )
        a_src = self.a[:self.out_features, :]   # [Fout, 1]
        a_dst = self.a[self.out_features:, :]   # [Fout, 1]
        wh1 = (h @ a_src).squeeze(-1)           # [N]
        wh2 = (h @ a_dst).squeeze(-1)           # [N]
        e = F.leaky_relu(wh1[row] + wh2[col], negative_slope=self.alpha)  # [E]

        # softmax per row (source node)
        attn = segment_softmax(e, row, num_segments=N)  # [E]
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # aggregate: out[i] = sum_j attn_ij * h[j]
        out = torch.zeros((N, self.out_features), device=h.device, dtype=h.dtype)
        out.scatter_add_(0, row.unsqueeze(-1).expand(-1, self.out_features),
                         attn.unsqueeze(-1) * h[col])

        return out


class MGCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.fusion_weights = conf['fusion_weights']

        self.modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight']).unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.init_emb()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph, self.uu_graph, self.bb_graph = raw_graph
        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

        self.gat_uu = GATLayer(self.embedding_size, self.embedding_size)
        self.gat_bb = GATLayer(self.embedding_size, self.embedding_size)

        self.mask_uu_graph = self.get_mask_graph(self.uu_graph)
        self.mask_bb_graph = self.get_mask_graph(self.bb_graph)

        self.getGAT_graph()

        # generate the graph without any dropouts for testing
        self.get_item_level_graph_ori()
        self.get_bundle_level_graph_ori()
        self.get_bundle_agg_graph_ori()

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.get_item_level_graph()
        self.get_bundle_level_graph()
        self.get_bundle_agg_graph()

        self.init_md_dropouts()

    def init_md_dropouts(self):
        self.item_level_dropout = nn.Dropout(self.conf["item_level_ratio"], True)
        self.bundle_level_dropout = nn.Dropout(self.conf["bundle_level_ratio"], True)
        self.bundle_agg_dropout = nn.Dropout(self.conf["bundle_agg_ratio"], True)

    def init_emb(self):
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def get_mask_graph(self, graph):
        graph = graph.tocoo()
        values = graph.data
        indices = np.vstack((graph.row, graph.col))
        dense_graph = torch.sparse.LongTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape)).to_dense()
        masked_graph = 1 - dense_graph

        return masked_graph

    def get_bundle_level_graph(self):
        ub_graph = self.ub_graph
        device = self.device
        modification_ratio = self.conf["bundle_level_ratio"]

        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bundle_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bundle_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.bundle_level_graph = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_bundle_level_graph_ori(self):
        ub_graph = self.ub_graph
        device = self.device
        bundle_level_graph = sp.bmat([[sp.csr_matrix((ub_graph.shape[0], ub_graph.shape[0])), ub_graph], [ub_graph.T, sp.csr_matrix((ub_graph.shape[1], ub_graph.shape[1]))]])
        self.bundle_level_graph_ori = to_tensor(laplace_transform(bundle_level_graph)).to(device)

    def get_item_level_graph(self):
        ui_graph = self.ui_graph
        device = self.device
        modification_ratio = self.conf["item_level_ratio"]

        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])
        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = item_level_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                item_level_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        self.item_level_graph = to_tensor(laplace_transform(item_level_graph)).to(device)

    def get_item_level_graph_ori(self):
        ui_graph = self.ui_graph
        device = self.device
        item_level_graph = sp.bmat([[sp.csr_matrix((ui_graph.shape[0], ui_graph.shape[0])), ui_graph], [ui_graph.T, sp.csr_matrix((ui_graph.shape[1], ui_graph.shape[1]))]])

        self.item_level_graph_ori = to_tensor(laplace_transform(item_level_graph)).to(device)

    def get_bundle_agg_graph(self):
        bi_graph = self.bi_graph
        device = self.device

        if self.conf["aug_type"] == "ED":
            modification_ratio = self.conf["bundle_agg_ratio"]
            graph = self.bi_graph.tocoo()
            values = np_edge_dropout(graph.data, modification_ratio)
            bi_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph = to_tensor(bi_graph).to(device)

    def get_bundle_agg_graph_ori(self):
        bi_graph = self.bi_graph
        device = self.device

        bundle_size = bi_graph.sum(axis=1) + 1e-8
        bi_graph = sp.diags(1/bundle_size.A.ravel()) @ bi_graph
        self.bundle_agg_graph_ori = to_tensor(bi_graph).to(device)


    def getGAT_graph(self):
        self.uu_indptr = torch.from_numpy(self.uu_graph.indptr).long().to(self.device)
        self.uu_indices = torch.from_numpy(self.uu_graph.indices).long().to(self.device)

        self.bb_indptr = torch.from_numpy(self.bb_graph.indptr).long().to(self.device)
        self.bb_indices = torch.from_numpy(self.bb_graph.indices).long().to(self.device)


    def one_propagate(self, graph, A_feature, B_feature, mess_dropout, test, num_layers):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test: # !!! important
                features = mess_dropout(features)

            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.mean(all_features, dim=1)

        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    def gat_one_propagate(self, feature, indptr, indice, graph_type, num_layers):
        all_features = [feature]

        for i in range(num_layers):
            if graph_type == "uu":
                feature = self.gat_uu(feature, indptr, indice)
            else:
                feature = self.gat_bb(feature, indptr, indice)

            all_features.append(F.normalize(feature, p=2, dim=1))

        all_features = torch.stack(all_features, 1)
        all_features = torch.mean(all_features, dim=1)
        return all_features


    def get_IL_bundle_rep(self, IL_items_feature, test):
        if test:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph_ori, IL_items_feature)
        else:
            IL_bundles_feature = torch.matmul(self.bundle_agg_graph, IL_items_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["bundle_agg_ratio"] != 0 and self.conf["aug_type"] == "MD" and not test:
            IL_bundles_feature = self.bundle_agg_dropout(IL_bundles_feature)

        return IL_bundles_feature


    def propagate(self, test=False):

        #  =============================  item level propagation  =============================
        if test:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph_ori, self.users_feature, self.items_feature, self.item_level_dropout, test, self.num_layers[0])
        else:
            IL_users_feature, IL_items_feature = self.one_propagate(self.item_level_graph, self.users_feature, self.items_feature, self.item_level_dropout, test, self.num_layers[0])
        # aggregate the items embeddings within one bundle to obtain the bundle representation
        IL_bundles_feature = self.get_IL_bundle_rep(IL_items_feature, test)
        #  ============================= bundle level propagation =============================
        if test:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph_ori, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test, self.num_layers[1])
        else:
            BL_users_feature, BL_bundles_feature = self.one_propagate(self.bundle_level_graph, self.users_feature, self.bundles_feature, self.bundle_level_dropout, test, self.num_layers[1])

        ML_users_feature = self.gat_one_propagate(self.users_feature, self.uu_indptr, self.uu_indices, "uu", self.num_layers[2])
        ML_bundles_feature = self.gat_one_propagate(self.bundles_feature, self.bb_indptr, self.bb_indices, "bb", self.num_layers[2])

        temp_users_feature = [IL_users_feature, BL_users_feature, ML_users_feature]
        temp_bundles_feature = [IL_bundles_feature, BL_bundles_feature, ML_bundles_feature]

        users_feature = torch.stack(temp_users_feature, dim=0)
        bundles_feature = torch.stack(temp_bundles_feature, dim=0)

        AL_users_feature = torch.sum(users_feature * self.modal_coefs, dim=0)
        AL_bundles_feature = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        users_output = [IL_users_feature, IL_items_feature, BL_users_feature, ML_users_feature, AL_users_feature]
        bundles_output = [IL_bundles_feature, BL_bundles_feature, ML_bundles_feature, AL_bundles_feature]

        return users_output, bundles_output


    def cal_c_loss(self, pos, aug, mask):
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1)
        ttl_score = torch.matmul(pos, aug.permute(1, 0))

        pos_score = torch.exp(pos_score / self.c_temp[0])
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp[0])*mask, axis=1)

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss

    def cal_ic_loss(self, emb, pos, neg):
        emb = emb[:, 0, :]

        emb = F.normalize(emb, p=2, dim=1)
        pos = F.normalize(pos, p=2, dim=-1)
        neg = F.normalize(neg, p=2, dim=-1)

        pos_score = torch.bmm(emb.unsqueeze(dim=1), pos.permute(0, 2, 1))
        pos_score = torch.sum(torch.exp(pos_score / self.c_temp[1]), axis=-1).squeeze()

        ttl_score = torch.bmm(emb.unsqueeze(dim=1), neg.permute(0, 2, 1))
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp[1]), axis=-1).squeeze()

        c_loss = - torch.mean(torch.log(pos_score / (pos_score + ttl_score)))

        return c_loss

    def cal_loss(self, users_feature, bundles_feature, mask_u, mask_b, u_ids, b_ids, pos_uu_samples, neg_uu_samples, pos_bb_samples, neg_bb_samples):
        # IL: item_level, BL: bundle_level
        # [bs, 1, emb_size]

        IL_users_feature, IL_items_feature, BL_users_feature, ML_users_feature, AL_users_feature = users_feature
        # [bs, 1+neg_num, emb_size]
        IL_bundles_feature, BL_bundles_feature, ML_bundles_feature, AL_bundles_feature = bundles_feature
        # [bs, 1+neg_num]
        pred = torch.sum(AL_users_feature * AL_bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        user_random_noise = torch.rand_like(u_ids, device=AL_users_feature.device)
        pos_uu_samples[:, 0, :] = pos_uu_samples[:, 0, :] + (torch.sign(u_ids) * F.normalize(user_random_noise, dim=-1) * 0.2).squeeze()

        bundle_random_noise = torch.rand_like(b_ids, device=AL_users_feature.device)
        pos_bb_samples[:, 0, :] = pos_bb_samples[:, 0, :] + (torch.sign(b_ids) * F.normalize(bundle_random_noise, dim=-1) * 0.2).squeeze()

        u_cross_view = self.cal_c_loss(IL_users_feature, BL_users_feature, mask_u) + self.cal_ic_loss(u_ids, pos_uu_samples, neg_uu_samples)
        b_cross_view = self.cal_c_loss(IL_bundles_feature, BL_bundles_feature, mask_b) + self.cal_ic_loss(b_ids, pos_bb_samples, neg_bb_samples)

        c_losses = [u_cross_view, b_cross_view]

        c_loss = sum(c_losses) / len(c_losses)

        return bpr_loss, c_loss


    def forward(self, batch):

        users, bundles, uid, bid, pos_uu_samples, neg_uu_samples, pos_bb_samples, neg_bb_samples = batch

        user_list = users.squeeze()
        bundle_list = bundles[:, 0]

        mask_u = self.mask_uu_graph[user_list.cpu()][:, user_list.cpu()].to(user_list.device)
        mask_b = self.mask_bb_graph[bundle_list.cpu()][:, bundle_list.cpu()].to(bundle_list.device)

        users_feature, bundles_feature = self.propagate()
        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]

        bundles_embedding = [i[bundles] for i in bundles_feature]

        u_ids = users_feature[-1][uid]
        b_ids = bundles_feature[-1][bid]
        pos_uu_samples = users_feature[-1][pos_uu_samples]
        neg_uu_samples = users_feature[-1][neg_uu_samples]
        pos_bb_samples = bundles_feature[-1][pos_bb_samples]
        neg_bb_samples = bundles_feature[-1][neg_bb_samples]


        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding, mask_u, mask_b, u_ids, b_ids, pos_uu_samples, neg_uu_samples, pos_bb_samples, neg_bb_samples)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        IL_users_feature, IL_items_feature, users_feature_atom, _, users_feature_non_atom = [i[users] for i in users_feature]
        IL_bundles_feature, bundles_feature_atom, _, bundles_feature_non_atom = bundles_feature

        scores = torch.mm(users_feature_non_atom, bundles_feature_non_atom.t())
        return scores
