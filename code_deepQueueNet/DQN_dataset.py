import torch.utils.data as data
import torch
import h5py

class DataFromH5File(data.Dataset):
    def __init__(self, filepath):
        h5File = h5py.File(filepath, 'r')
        self.x = h5File['x']
        self.y = h5File['y']
        
    def __getitem__(self, idx):
        label = torch.from_numpy(self.y[idx]).float()
        data = torch.from_numpy(self.x[idx]).float()
        return data, label
    
    def __len__(self):
        assert self.y.shape[0] == self.x.shape[0], "Wrong data length"
        return self.y.shape[0]
