from torch.utils import data
import numpy as np
import random
import torch

import glob
import fnmatch

import sys
sys.path.append(".")
from OCT2SysD_Tools import readBESClinicalCsv

class OCT2SysD_DataSet(data.Dataset):
    def __init__(self, mode, hps=None, transform=None):
        '''

        :param mode: training, validation, test
        :param hps:
        :param transform:
        '''
        super().__init__()
        self.m_mode = mode
        self.hps = hps

        if mode == "training":
            IDPath = hps.trainingDataPath
        elif mode == "validation":
            IDPath = hps.validationDataPath
        elif mode == "test":
            IDPath = hps.testDataPath
        else:
            print(f"OCT2SysDiseaseDataSet mode error")
            assert False

        # read GT
        self.m_labels = readBESClinicalCsv(hps.GTPath)

        with open(IDPath, 'r') as idFile:
            IDList = idFile.readlines()
        IDList = [item[0:-1] for item in IDList]  # erase '\n'

        self.m_transform = transform

        # get all correct volume numpy path
        volumeList = glob.glob(hps.dataDir + "/*_"+hps.ODOS+"_*_Volume.npy")
        nonexistIDList = []

        # make sure ID and volume has strict corresponding order
        self.m_volumesPath = []
        self.m_IDs = []
        for ID in IDList:
            resultList = fnmatch.filter(volumeList, "*/" + ID + "_" + hps.ODOS+"_*_Volume.npy")
            length = len(resultList)
            if 0 == length:
                nonexistIDList.append(ID)
            elif length > 1:
                print(f"Mulitple ID files: {resultList}")
            else:
                self.m_volumesPath.append(resultList[0])
                self.m_IDs.append(ID)
        if len(nonexistIDList) > 0:
            print(f"Error:  nonexistIDList:\n {nonexistIDList}")
            assert False


    def __len__(self):
        return len(self.m_volumesPath)

    def getGTDict(self):
        return self.m_labels

    def addVolumeGradient(self, volume):
        '''
        gradient should use both-side gradient approximation formula: (f_{i+1}-f_{i-1})/2,
        and at boundaries of images, use single-side gradient approximation formula: (f_{i}- f_{i-1})/2
        :param volume: in size: SxHxW
        :param gradChannels:  integer
        :return:
                Bx3xHxW, added gradient volume without normalization
        '''
        S,H,W =volume.shape

        image_1H = volume[:, 0:-2, :]  # size: S, H-2,W
        image1H = volume[:, 2:, :]
        gradH = torch.cat(((volume[:, 1, :] - volume[:, 0, :]).view(S, 1, W),
                           (image1H - image_1H)/2.0,
                           (volume[:, -1, :] - volume[:, -2, :]).view(S, 1, W)), dim=1)  # size: S, H,W; grad90

        image_1W = volume[:,:, 0:-2]  # size: S, H,W-2
        image1W = volume[:, :, 2:]
        gradW = torch.cat(((volume[:,:, 1] - volume[:,:, 0]).view(S, H, 1),
                           (image1W - image_1W)/2,
                           (volume[:,:, -1] - volume[:,:, -2]).view(S, H, 1)), dim=2)  # size: S, H,W; grad0

        # concatenate
        gradVolume = torch.cat((volume.unsqueeze(dim=1), gradH.unsqueeze(dim=1), gradW.unsqueeze(dim=1)), dim=1) # B,3,H,W

        return gradVolume


    def __getitem__(self, index):
        epsilon = 1e-8
        ID = self.m_IDs[index]

        labels= []
        if self.hps.existGTLabel:
            labels = self.m_labels[int(ID)]

        volumePath = self.m_volumesPath[index]
        npVolume = np.load(volumePath)

        data = torch.from_numpy(npVolume).to(device=self.hps.device, dtype=torch.float32)
        S,H,W = data.shape

        # transform for data augmentation
        if self.m_transform:
            data = self.m_transform(data)  # size: SxHxW

        if 0 != self.hps.gradChannels:
            data = self.addVolumeGradient(data)  # S,3,H,W
        else:
            data = data.unsqueeze(dim=1)  # S,1,H,W

        # normalization before output to dataloader
        # AlexNex, GoogleNet V1, VGG, ResNet only do mean subtraction without dividing std.
        #mean = torch.mean(data, dim=(-1, -2), keepdim=True)
        #mean = mean.expand_as(data)
        #data = data - mean

        # Normalization
        std, mean = torch.std_mean(data, dim=(-1, -2), keepdim=True)
        std = std.expand_as(data)
        mean = mean.expand_as(data)
        data = (data - mean) / (std + epsilon)  # size: Sx3xHxW, or S,1,H,W

        result = {"images": data,  # S,3,H,W or S,1,H,W
                  "GTs": labels,
                  "IDs": ID
                 }
        return result  # B,S,3,H,W
        # output need to merge th B,S dimension into one dimension.









