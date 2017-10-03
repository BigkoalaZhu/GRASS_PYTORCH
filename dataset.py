import torch
from torch.utils import data
from scipy.io import loadmat

class GRASS(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        o = torch.from_numpy(loadmat(root+'/ops.mat')['ops']).int()
        b = torch.from_numpy(loadmat(root+'/boxes.mat')['boxes']).float()
        s = torch.from_numpy(loadmat(root+'/syms.mat')['syms']).float()
        w = torch.from_numpy(loadmat(root+'/weights.mat')['weights']).float()
        l = o.size()[1]
        self.opData = torch.chunk(o,l,1)
        self.boxData = torch.chunk(b,l,1)
        self.symData = torch.chunk(s,l,1)
        self.wData = torch.chunk(w,l,1)
        self.transform = transform

    def __getitem__(self, index):

        box = torch.t(self.boxData[index])
        op = torch.t(self.opData[index])
        sym = torch.t(self.symData[index])
        w = torch.t(self.wData[index])

        return box, op, sym, w

    def __len__(self):
        return len(self.opData)