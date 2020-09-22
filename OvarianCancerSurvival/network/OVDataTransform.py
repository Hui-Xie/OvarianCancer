# OCT data transform class

import random
import torch
import math

class OVDataTransform(object):
    def __init__(self, hps):
        '''
        without random slice.
        '''
        self.hps = hps
        self.m_edgeCropRate = math.sqrt(hps.randomCropArea)


    def __call__(self, inputData):
        '''

        :param inputData:  a Tensor of size(S, H,W),
        :return:
                a unnormalized tensor of size: S,H,W
        '''

        device = inputData.device
        S,H,W = inputData.shape
        newH = int(H * self.m_edgeCropRate)
        newW = int(W * self.m_edgeCropRate)
        gapH = H- newH
        gapW = W -newW

        # transformed data
        tfData = torch.empty((S,newH,newW), device=device,dtype=torch.float32)

        for i in range(S):
            data = inputData[i,:,:]

            # random crop
            startH = random.randrange(0,gapH)
            startW = random.randrange(0,gapW)
            data = data[startH:startH+newH, startW: startW+newW]

            # flip
            if random.uniform(0, 1) < self.hps.flipProb:
                data = torch.flip(data, [1])  # flip horizontal
            if random.uniform(0, 1) < self.hps.flipProb:
                data = torch.flip(data, [0])  # flip vertical

            # gaussian noise
            if random.uniform(0, 1) < self.hps.augmentProb:
                data = data + torch.normal(0.0, self.hps.gaussianNoiseStd, size=data.size()).to(device=device,dtype=torch.float)

            # salt-pepper noise
            if random.uniform(0, 1) < self.hps.augmentProb:
                # salt: maxValue; pepper: minValue
                mask = torch.empty(data.size(),dtype=torch.float,device=device).uniform_(0,1)
                pepperMask = (mask <= self.hps.saltPepperRate)
                saltMask = (mask <= self.hps.saltPepperRate*self.hps.saltRate)
                pepperMask ^= saltMask  # xor
                max = data.max()
                min = data.min()
                data[torch.nonzero(pepperMask, as_tuple=True)] = min
                data[torch.nonzero(saltMask,   as_tuple=True)] = max

            tfData[i, :, :] = data

        return tfData

    def __repr__(self):
        return self.__class__.__name__

class OVDataTransform_RandomSlice(object):
    def __init__(self, hps):
        self.hps = hps
        self.m_edgeCropRate = math.sqrt(hps.randomCropArea)


    def __call__(self, inputData):
        '''

        :param inputData:  a Tensor of size(S, H,W),
        :return:
                a unnormalized tensor of size: S,H,W
        '''

        device = inputData.device
        S,H,W = inputData.shape
        newS = int(S* self.hps.randomSlicesRate)
        newSList = random.sample(list(range(0,S)), newS)
        newH = int(H * self.m_edgeCropRate)
        newW = int(W * self.m_edgeCropRate)
        gapH = H- newH
        gapW = W -newW

        # transformed data
        tfData = torch.empty((newS,newH,newW), device=device,dtype=torch.float32)

        for i, s in enumerate(newSList):
            data = inputData[s,:,:]

            # random crop
            startH = random.randrange(0,gapH)
            startW = random.randrange(0,gapW)
            data = data[startH:startH+newH, startW: startW+newW]

            # flip
            if random.uniform(0, 1) < self.hps.flipProb:
                data = torch.flip(data, [1])  # flip horizontal
            if random.uniform(0, 1) < self.hps.flipProb:
                data = torch.flip(data, [0])  # flip vertical

            # gaussian noise
            if random.uniform(0, 1) < self.hps.augmentProb:
                data = data + torch.normal(0.0, self.hps.gaussianNoiseStd, size=data.size()).to(device=device,dtype=torch.float)

            # salt-pepper noise
            if random.uniform(0, 1) < self.hps.augmentProb:
                # salt: maxValue; pepper: minValue
                mask = torch.empty(data.size(),dtype=torch.float,device=device).uniform_(0,1)
                pepperMask = (mask <= self.hps.saltPepperRate)
                saltMask = (mask <= self.hps.saltPepperRate*self.hps.saltRate)
                pepperMask ^= saltMask  # xor
                max = data.max()
                min = data.min()
                data[torch.nonzero(pepperMask, as_tuple=True)] = min
                data[torch.nonzero(saltMask,   as_tuple=True)] = max

            tfData[i, :, :] = data

        return tfData

    def __repr__(self):
        return self.__class__.__name__