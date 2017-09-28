import torch
from torch.utils import data
from scipy.io import loadmat

class GRASS(data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.data = loadmat(root)['data'][0]
        self.transform = transform

    def __getitem__(self, index):

        boxes = torch.from_numpy(self.data[index]['boxes'][0][0]).float()
        symshapes = torch.from_numpy(self.data[index]['symshapes'][0][0]).float()
        treekids = torch.from_numpy(self.data[index]['treekids'][0][0]).int()

        s = treekids.size()
        symparams = torch.zeros(s[0],8)
        for ii in range(s[0]):
            if  self.data[index]['symparams'][0][0][0][ii].shape[0] != 0:
                symparams[ii,:] = torch.from_numpy(self.data[index]['symparams'][0][0][0][ii])

        sample = {'boxes': boxes, 'symshapes': symshapes, 'treekids': treekids, 'symparams': symparams}

        if self.transform:
            sample = self.transform(sample)

        return boxes,symshapes,treekids,symparams

    def __len__(self):
        return len(self.data)