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
                 TTA=False, TTA_Degree=0, scaleNumerator=1, scaleDenominator=1, gradChannels=0):
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
        self.m_gradChannels = gradChannels

    def __len__(self):
        return self.m_images.size()[0]

    def generateGradientImage(self, image, gradChannels):
        H,W =image.shape
        device = image.device
        image0H = image[0:-1,:]  # size: H-1,W
        image1H = image[1:,  :]
        gradH   = image1H-image0H
        gradH = torch.cat((gradH, torch.zeros((1, W), device=device)), dim=0)  # size: H,W

        image0W = image[:,0:-1]  # size: H,W-1
        image1W = image[:,1:  ]
        gradW = image1W - image0W
        gradW = torch.cat((gradW, torch.zeros((H, 1), device=device)), dim=1)  # size: H,W

        if 1 == gradChannels:
            return torch.sqrt(torch.pow(gradH,2)+torch.pow(gradW,2))
        elif 2 == gradChannels:
            return gradH, gradW,
        elif 3 == gradChannels:
            onesHW = torch.ones_like(image)
            negOnesHW = -onesHW
            signHW = torch.where(gradH*gradW >= 0, onesHW, negOnesHW)

            e = 1e-8
            gradDirection = torch.atan(signHW*(torch.abs(gradH)+e)/(torch.abs(gradW)+e))
            return gradH, gradW, gradDirection
        else:
            print(f"Currently do not support gradChannels >3")
            assert False
            return None


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
        image = data.unsqueeze(dim=0)
        if 0 != self.m_gradChannels:
            grads = self.generateGradientImage(data, self.m_gradChannels)
            for grad in grads:
                image = torch.cat((image, grad.unsqueeze(dim=0)),dim=0)

        result = {"images": image,
                  "GTs": label,
                  "gaussianGTs": [] if 0 == self.m_sigma else gaussianizeLabels(label, self.m_sigma, H),
                  "IDs": self.m_IDs[str(index)],
                  "layers": getLayerLabels(label,H) }
        return result









