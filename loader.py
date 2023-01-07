import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, item):
        data = torch.FloatTensor(self.data[item].reshape(4, -1))
        label = torch.FloatTensor(self.label[item])
        return data, label

    def __len__(self):
        return len(self.data)
