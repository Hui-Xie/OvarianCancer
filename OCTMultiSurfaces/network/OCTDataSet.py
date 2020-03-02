from torch.utils import data
import numpy as np
import json
import torch
import torchvision.transforms as TF

import sys
sys.path.append(".")
from OCTAugmentation import *


class OCTDataSet(data.Dataset):
    def __init__(self, imagesPath, labelPath, IDPath, transform=None, device=None, sigma=20.0, lacingWidth=0,
                 TTA=False, TTA_Degree=0, scaleNumerator=1, scaleDenominator=1):
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
        self.m_TTA = TTA
        self.m_TTA_Degree = TTA_Degree
        self.m_scaleNumerator = scaleNumerator
        self.m_scaleDenominator = scaleDenominator

    def __len__(self):
        return self.m_images.size()[0]

    def __getitem__(self, index):
        data = self.m_images[index,]
        label = self.m_labels[index,]
        if self.m_transform:
            data, label = self.m_transform(data, label)

        if self.m_TTA and 0 != self.m_TTA_Degree:
            data, label = polarImageLabelRotate_Tensor(data, label, rotation=self.m_TTA_Degree)

        if 0 != self.m_lacingWidth:
            data, label = lacePolarImageLabel(data,label,self.m_lacingWidth)

        if 1 != self.m_scaleNumerator or 1 != self.m_scaleDenominator:  # this will change the Height of polar image
            data = scalePolarImage(data, self.m_scaleNumerator, self.m_scaleDenominator)
            label = scalePolarLabel(label, self.m_scaleNumerator, self.m_scaleDenominator)

        H, W = data.shape
        result = {"images": data.unsqueeze(dim=0),
                  "GTs": label,
                  "gaussianGTs": [] if 0 == self.m_sigma else gaussianizeLabels(label, self.m_sigma, H),
                  "IDs": self.m_IDs[str(index)],
                  "layers": getLayerLabels(label,H) }
        return result









