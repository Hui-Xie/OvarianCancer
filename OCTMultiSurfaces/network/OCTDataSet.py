from torch.utils import data
import numpy as np
import json
import math
import torch



class OCTDataSet(data.Dataset):
    def __init__(self, imagesPath, labelPath, IDPath, transform=None, device=None, sigma=20):
        self.m_device = device
        self.m_sigma = sigma
        self.m_images = torch.from_numpy(np.load(imagesPath).astype(np.float32)).to(self.m_device)  # slice, H, W

        self.m_labels = torch.from_numpy(np.load(labelPath).astype(np.float32)).to(self.m_device)  # slice, num_surface, W
        with open(IDPath) as f:
            self.m_IDs = json.load(f)
        self.m_transform = transform


    def __len__(self):
        return self.m_images.size()[0]

    def __getitem__(self, index):
        S, H, W = self.m_images.shape
        result = {"image": self.m_images[index,].unsqueeze(dim=0),
                  "GT": self.m_labels[index,],
                  "gaussianGT": self.gaussianizeLabels(self.m_labels[index,], self.m_sigma, H),
                  "ID": self.m_IDs[str(index)]}
        return result

    def gaussianizeLabels(self, rawLabels, sigma, H):
        '''
        input: tensor(slice, Num_surface, W)
        output: tensor(slice, Num_surace, H, W)
        '''
        Mu = rawLabels
        Num_Surface, W = Mu.shape
        Mu = Mu.unsqueeze(dim=-2)
        Mu = Mu.expand((Num_Surface,H,W))
        X = torch.arange(start=0.0, end=H, step=1.0).view((H,1)).to(self.m_device)
        X = X.expand((Num_Surface,H,W))
        pi = math.pi
        G  = torch.exp(-(X-Mu)**2/(2*sigma*sigma))/(sigma*math.sqrt(2*pi))
        return G





