# Dataset 

## 3ET dataset
[dataloader](https://github.com/qinche106/cb-convlstm-eyetracking#tonic-dataloader)

## Install
```bash
pip install tonic --pre
pip install torch
```

## Download dataset
### Manually (Recommended)
- [Download](https://dl.dropboxusercontent.com/s/1hyer8egd8843t9/ThreeET_Eyetracking.zip?dl=0)
- In the folder of `three_et.py`
  - ```bash
    mkdir data
    ```
  - extract the files to `./data`

### Automatically
- usage
  ```python
  ThreeET(download=True)
  ```

## Usage
```python
from dataset import ThreeET
data = ThreeET()
# data.train: torch.utils.data.Dataset
# data.test: torch.utils.data.Dataset
```