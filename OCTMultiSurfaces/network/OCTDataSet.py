from torch.utils import data
import numpy as np
import json
import torch
import torchvision.transforms as TF

import sys
sys.path.append(".")
from OCTAugmentation import *


class OCTDataSet(data.Dataset):
    def __init__(self, imagesPath, IDPath=None, labelPath=None, transform=None, hps=None):
        self.hps = hps

        # image uses float32
        images = torch.from_numpy(np.load(imagesPath).astype(np.float32)).to(self.hps.device, dtype=torch.float)  # slice, H, W
        # normalize images for each slice
        std,mean = torch.std_mean(images, dim=(1,2))
        self.m_images = TF.Normalize(mean, std)(images)

        if labelPath is not None:
            self.m_labels = torch.from_numpy(np.load(labelPath).astype(np.float32)).to(self.hps.device, dtype=torch.float)  # slice, num_surface, W
        else:
            self.m_labels = None

        with open(IDPath) as f:
            self.m_IDs = json.load(f)
        self.m_transform = transform


    def __len__(self):
        return self.m_images.size()[0]

    def generateGradientImage(self, image, gradChannels):
        H,W =image.shape
        device = image.device
        image0H = image[0:-1,:]  # size: H-1,W
        image1H = image[1:,  :]
        gradH   = image1H-image0H
        gradH = torch.cat((gradH, torch.zeros((1, W), device=device)), dim=0)  # size: H,W; grad90

        image0W = image[:,0:-1]  # size: H,W-1
        image1W = image[:,1:  ]
        gradW = image1W - image0W
        gradW = torch.cat((gradW, torch.zeros((H, 1), device=device)), dim=1)  # size: H,W; grad0

        gradMagnitudeHW = torch.sqrt(torch.pow(gradH,2)+torch.pow(gradW,2))

        if gradChannels>=3:
            onesImage = torch.ones_like(image)
            negOnesImage = -onesImage
            signHW = torch.where(gradH * gradW >= 0, onesImage, negOnesImage)
            e = 1e-8
            gradDirectionHW = torch.atan(signHW * torch.abs(gradH) / (torch.abs(gradW) + e))

        if gradChannels >= 5:
            image45_0 = image[0:-1,1:]  # size: H-1,W-1
            image45_1 = image[1:,0:-1]  # size: H-1,W-1
            grad45 = image45_1 - image45_0 # size: H-1,W-1
            grad45 = torch.cat((torch.zeros((H-1,1), device=device), grad45), dim=1)
            grad45 = torch.cat((grad45, torch.zeros((1, W), device=device)), dim=0)

            image135_0 = image[0:-1, 0:-1]  # size: H-1,W-1
            image135_1 = image[1:, 1:]  # size: H-1,W-1
            grad135 = image135_1 - image135_0  # size: H-1,W-1
            grad135 = torch.cat((grad135, torch.zeros((H - 1, 1), device=device)), dim=1)
            grad135 = torch.cat((grad135, torch.zeros((1, W), device=device)), dim=0)

        if gradChannels >= 7:
            sign135_45 = torch.where(grad135 * grad45 >= 0, onesImage, negOnesImage)
            gradDirection135_45 = torch.atan(sign135_45 * torch.abs(grad135) / (torch.abs(grad45) + e))

        if 1 == gradChannels:
            return gradMagnitudeHW
        elif 2 == gradChannels:
            return gradH, gradW,
        elif 3 == gradChannels:
             return gradH, gradW, gradDirectionHW
        elif 4 == gradChannels:
            return gradH, gradW, gradMagnitudeHW, gradDirectionHW
        elif 5 == gradChannels:
            return gradH, gradW, grad45, grad135, gradDirectionHW
        elif 6 == gradChannels:
            return gradH, gradW, grad45, grad135, gradMagnitudeHW, gradDirectionHW
        elif 7 == gradChannels:
            return gradH, gradW, grad45, grad135, gradMagnitudeHW, gradDirectionHW, gradDirection135_45
        else:
            print(f"Currently do not support gradChannels >7")
            assert False
            return None


    def __getitem__(self, index):
        data = self.m_images[index,]
        if self.m_labels is not None:
            label = self.m_labels[index,] # size: N,W
        else:
            label = None

        if self.m_transform:
            data, label = self.m_transform(data, label)

        if self.hps.TTA and 0 != self.hps.TTA_Degree:
            data, label = polarImageLabelRotate_Tensor(data, label, rotation=self.hps.TTA_Degree)

        if 0 != self.hps.lacingWidth:
            data, label = lacePolarImageLabel(data,label,self.hps.lacingWidth)

        if 1 != self.hps.scaleNumerator or 1 != self.hps.scaleDenominator:  # this will change the Height of polar image
            data = scalePolarImage(data, self.hps.scaleNumerator, self.hps.scaleDenominator)
            label = scalePolarLabel(label, self.hps.scaleNumerator, self.hps.scaleDenominator)

        H, W = data.shape
        N, W1 = label.shape
        assert W==W1
        image = data.unsqueeze(dim=0)
        if 0 != self.hps.gradChannels:
            grads = self.generateGradientImage(data, self.hps.gradChannels)
            for grad in grads:
                image = torch.cat((image, grad.unsqueeze(dim=0)),dim=0)

        riftWidthGT = []
        if hasattr(self.hps, 'useRiftWidth') and True == self.hps.useRiftWidth:
            riftWidthGT = torch.cat((label[0,:].unsqueeze(dim=0),label[1:,:]-label[0:-1,:]),dim=0)



        result = {"images": image,
                  "GTs": [] if label is None else label,
                  "gaussianGTs": [] if 0 == self.hps.sigma or label is None  else gaussianizeLabels(label, self.hps.sigma, H),
                  "IDs": self.m_IDs[str(index)],
                  "layers": [] if label is None else getLayerLabels(label,H),
                  "riftWidth": riftWidthGT}
        return result









