import torch
from torch.utils.data import Dataset

use_cuda = torch.cuda.is_available()
print('use_cuda:', use_cuda)
device = torch.device('cuda:0' if use_cuda else 'cpu')


class eegDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor.to(device)
        self.y = y_tensor.to(device)
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)
