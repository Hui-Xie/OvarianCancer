from torch.utils import data
import numpy as np
import json
import torch
import torchvision.transforms as TF

import sys
sys.path.append(".")
from OCTAugmentation import *


class OCTDataSet(data.Dataset):
    def __init__(self, imagesPath, labelPath, IDPath, transform=None, device=None, sigma=20.0, lacingWidth=0):
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
        self.m_lacingWidth = lacingWidth

    def __len__(self):
        return self.m_images.size()[0]

    def __getitem__(self, index):
        S, H, W = self.m_images.shape
        data = self.m_images[index,]
        label = self.m_labels[index,]
        if self.m_transform:
            data, label = self.m_transform(data, label)
        if 0 != self.m_lacingWidth:
            data, label = lacePolarImageLabel(data,label,self.m_lacingWidth)

        result = {"images": data.unsqueeze(dim=0),
                  "GTs": label,
                  "gaussianGTs": gaussianizeLabels(label, self.m_sigma, H),
                  "IDs": self.m_IDs[str(index)],
                  "layers": getLayerLabels(label,H) }
        return result









