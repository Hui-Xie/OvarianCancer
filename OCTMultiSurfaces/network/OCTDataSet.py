from torch.utils import data
import numpy as np
import json
import math
import torch
import torchvision.transforms as TF

class OCTDataSet(data.Dataset):
    def __init__(self, imagesPath, labelPath, IDPath, transform=None, device=None, sigma=20.0):
        self.m_device = device
        self.m_sigma = sigma

        # image uses float32
        images = torch.from_numpy(np.load(imagesPath).astype(np.float32)).to(self.m_device, dtype=torch.float)  # slice, H, W
        # normalize images for each slice
        std,mean = torch.std_mean(images, dim=(1,2))
        self.m_images = TF.Normalize(mean, std)(images)

        self.m_labels = torch.from_numpy(np.load(labelPath).astype(np.float32)).to(self.m_device, dtype=torch.float)  # slice, num_surface, W
        with open(IDPath) as f:
            self.m_IDs = json.load(f)
        self.m_transform = transform


    def __len__(self):
        return self.m_images.size()[0]

    def __getitem__(self, index):
        S, H, W = self.m_images.shape
        data = self.m_images[index,]
        if self.m_transform:
            data = self.m_transform(data)
        result = {"images": data.unsqueeze(dim=0),
                  "GTs": self.m_labels[index,],
                  "gaussianGTs": self.gaussianizeLabels(self.m_labels[index,], self.m_sigma, H),
                  "IDs": self.m_IDs[str(index)]}
        return result

    def gaussianizeLabels(self, rawLabels, sigma, H):
        '''
        input: tensor(Num_surface, W)
        output: tensor(Num_surace, H, W)
        '''
        Mu = rawLabels
        Num_Surface, W = Mu.shape
        Mu = Mu.unsqueeze(dim=-2)
        Mu = Mu.expand((Num_Surface,H,W))
        X = torch.arange(start=0.0, end=H, step=1.0).view((H,1)).to(self.m_device, dtype=torch.float32)
        X = X.expand((Num_Surface,H,W))
        pi = math.pi
        G  = torch.exp(-(X-Mu)**2/(2*sigma*sigma))/(sigma*math.sqrt(2*pi))
        return G

    def getLayersLabel(self, surfaceLabels, height):
        '''

        :param surfaceLabels: float tensor in size of (N,W) where N is the number of surfaces.
                height: original image height
        :return: layerLabels: long tensor in size of (H, W) in which each element is long integer  of [0,N] indicating belonging layer

        '''
        H = height
        device = surfaceLabels.device
        N,W = surfaceLabels.shape # N is the number of surface
        layerLabels = torch.zeros((H, W), dtype=torch.long, device=device)
        surfaceLabels = (surfaceLabels + 0.5).long() # let surface height match grid
        surfacesCode = torch.tensor(range(1,N+1)).unsqueeze(dim=-1).expand_as(surfaceLabels)
        layerLabels = torch.scatter(0, surfaceLabels, surfacesCode)
        curveLoc = surfaceLabels[0,:]+1
        bottom = torch.ones((1,W),  dtype=torch.long, device=device)*H
        zeroWidth = torch.zeros((1,W), dtype=torch.long, device=device)
        while torch.any(curveLoc < bottom):
            layerLabels[curveLoc,:] = torch.where(zeroWidth == layerLabels[curveLoc,:],  layerLabels[curveLoc-1,:], layerLabels[curveLoc,:])
            curveLoc = torch.where(curveLoc < bottom, curveLoc+1, curveLoc)








