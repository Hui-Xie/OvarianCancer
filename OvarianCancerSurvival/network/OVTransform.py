# OCT data transform class

import random
import torch
import torchvision.transforms as TF
import sys
from network.OCTAugmentation import *

class OVDataTransform(object):
    def __init__(self, hps):
        self.m_prob = hps.augmentProb
        self.m_noiseStd = hps.gaussianNoiseStd
        self.m_filpProb = hps.flipProb
        self.m_saltPepperRate = hps.saltPepperRate
        self.m_saltRate = hps.saltRate


    def __call__(self, inputData):
        '''

        :param inputData:  a Tensor of size(S, H,W),
        :return:
                a normalized tensor of size: S,H,W
        '''
        H,W = inputData.shape
        device = inputData.device
        dirt =False
        data = inputData.clone()
        if inputLabel is not None:
            label = inputLabel.clone()
        else:
            label = None

        # gaussian noise
        if random.uniform(0, 1) < self.m_prob:
            data = data + torch.normal(0.0, self.m_noiseStd, size=data.size()).to(device=device,dtype=torch.float)
            dirt = True

        # salt-pepper noise
        if random.uniform(0, 1) < self.m_prob:
            # salt: maxValue; pepper: minValue
            mask = torch.empty(data.size(),dtype=torch.float,device=device).uniform_(0,1)
            pepperMask = mask <= self.m_saltPepperRate
            saltMask = mask <= self.m_saltPepperRate*self.m_saltRate
            pepperMask ^= saltMask
            max = data.max()
            min = data.min()
            data[torch.nonzero(pepperMask, as_tuple=True)] = min
            data[torch.nonzero(saltMask,   as_tuple=True)] = max
            dirt =True

        # rotation
        if self.m_rotation and inputLabel is not None:
            rotation = random.randint(0,359)
            data, label = polarImageLabelRotate_Tensor(data, label, rotation=rotation)

        # normalize again
        if dirt:
            std, mean = torch.std_mean(data)
            data = TF.Normalize([mean], [std])(data.unsqueeze(dim=0))
            data = data.squeeze(dim=0)

        if inputLabel is None:
            return data
        else:
            return data, label

    def __repr__(self):
        return self.__class__.__name__