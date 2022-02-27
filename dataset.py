import numpy as np

import torch
from torch.utils.data.dataset import Dataset
import scipy.io as sio
import h5py

class BITDataset(Dataset):
    def __init__(self, f):
        super(BITDataset, self).__init__()
        self.data = None
        self.label = None
        self.path = f
        with h5py.File(self.path, 'r') as file:
            self.len = file.get('data').shape[0]

    def __getitem__(self, idx):
        if self.data == None:
            self.data = h5py.File(self.path, 'r').get('data')
            self.label = h5py.File(self.path, 'r').get('label')
        patch = torch.from_numpy(np.transpose(self.data[idx], (2, 0, 1))).float()
        label = self.label[idx]
        return patch, label

    def __len__(self):
        return self.len

    def __labels__(self):
        return self.label


class Dataset(Dataset):
    def __init__(self, dataset, transfor):
        self.data = dataset[0].astype(np.float32)
        self.transformer = transfor
        self.labels = []
        for n in dataset[1]: self.labels += [int(n)]

    def __getitem__(self, index):
        img = torch.from_numpy(np.asarray(self.data[index,:,:,:]))
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.labels)

    def __labels__(self):
        return self.labels
