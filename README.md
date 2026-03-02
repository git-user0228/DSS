## Requirements

- **OS**: Ubuntu 18.04 or higher  
- **Python**: 3.7.3 or above  
- **Supported (tested) CUDA versions**: 10.2  
- **PyTorch**: 1.9.0 or above

## Code Structure

- **Entry script for training and evaluation**: `train.py`
- **Configuration file**: `config.yaml`
- **Data preprocessing and dataloader**: `utility.py`
- **Model definitions**: `./models/`
- **TensorBoard logs**: `./runs/`
- **Text-format logs**: `./log/`
- **Saved checkpoints** (best model and associated config for each experiment): `./checkpoints/`

## Note
- Due to upload and storage limitations, the dataset used in this paper can be downloaded from [here](https://drive.google.com/file/d/1zGYmC67tini5rO7HdXAlCN3d7bILZO8Y/view?usp=sharing).Specific modality information can be obtained from the POG dataset or the Steam API using the item_id_map provided in the dataset.
