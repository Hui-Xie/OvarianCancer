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
        sliceList = glob.glob(hps.dataDir + "/*_*_*_Slice??.npy")
        nonexistIDList = []

        # make sure slice ID and slice has strict corresponding order
        self.m_slicesPath = []
        self.m_volumeIDs = []
        self.m_sliceIDs = []  # slice IDs are repeated volume ID for many slice in a volume
        self.m_volumeStartIndex = []
        self.m_volumeNumSlices = []
        # slices of [volumeStartIndex[i]:volumeStartIndex[i]+volumeNumSlices[i]) belong to a same volume.
        for i,ID in enumerate(IDList):
            resultList = fnmatch.filter(sliceList, "*/" + ID + "_*_*_Slice??.npy")
            resultList.sort()
            numSlices = len(resultList)
            if 0 == numSlices:
                nonexistIDList.append(ID)
            else:
                if 0 == i:
                    self.m_volumeStartIndex.append(0)
                else:
                    self.m_volumeStartIndex.append(self.m_volumeStartIndex[i-1] + self.m_volumeNumSlices[i-1])
                self.m_volumeNumSlices.append(numSlices)

                self.m_volumeIDs.append(ID)
                for n in range(numSlices):
                    self.m_slicesPath.append(resultList[n])
                    self.m_sliceIDs.append(ID)

        if len(nonexistIDList) > 0:
            print(f"Error:  nonexistIDList:\n {nonexistIDList}")
            assert False

    def __len__(self):
        return len(self.m_volumeIDs)

    def getGTDict(self):
        return self.m_labels

    def addSliceGradient(self, slice):
        '''
        gradient should use both-side gradient approximation formula: (f_{i+1}-f_{i-1})/2,
        and at boundaries of images, use single-side gradient approximation formula: (f_{i}- f_{i-1})/2
        :param slice: in size: HxW
        :return:
                3xHxW, added gradient volume without normalization
        '''
        H,W =slice.shape

        image_1H = slice[0:-2, :]  # size: H-2,W
        image1H = slice[2:, :]
        gradH = torch.cat(((slice[1, :] - slice[0, :]).view(1, W),
                           (image1H - image_1H)/2.0,
                           (slice[-1, :] - slice[-2, :]).view(1, W)), dim=0)  # size: H,W; grad90

        image_1W = slice[:, 0:-2]  # size: H,W-2
        image1W = slice[:, 2:]
        gradW = torch.cat(((slice[:, 1] - slice[:, 0]).view(H, 1),
                           (image1W - image_1W)/2,
                           (slice[:, -1] - slice[:, -2]).view(H, 1)), dim=1)  # size: H,W; grad0

        # concatenate
        gradVolume = torch.cat((slice.unsqueeze(dim=0), gradH.unsqueeze(dim=0), gradW.unsqueeze(dim=0)), dim=0) # 3,H,W

        return gradVolume


    def __getitem__(self, index):
        if self.m_mode == "training":
            return self.getSliceItem(index)
        elif (self.m_mode == "validation") or (self.m_mode == "test"):
            return self.getVolumeItem(index)
        else:
            print("dataset model error.")
            assert False


    def getSliceItem(self, index):
        startIndex = self.m_volumeStartIndex[index]
        numSlices = self.m_volumeNumSlices[index]
        sliceIndex = random.randrange(startIndex, startIndex + numSlices)

        epsilon = 1e-8
        ID = self.m_sliceIDs[sliceIndex]

        labels = []
        if self.hps.existGTLabel:
            labels = self.m_labels[int(ID)]

        slicePath = self.m_slicesPath[sliceIndex]
        npSlice = np.load(slicePath)

        data = torch.from_numpy(npSlice).to(device=self.hps.device, dtype=torch.float32)
        H, W = data.shape

        # transform for data augmentation
        if self.m_transform:
            data = self.m_transform(data)  # size: HxW

        if 0 != self.hps.gradChannels:
            data = self.addSliceGradient(data)  # 3,H,W
        else:
            data = data.unsqueeze(dim=0)  # 1,H,W

        # normalization before output to dataloader
        # AlexNex, GoogleNet V1, VGG, ResNet only do mean subtraction without dividing std.
        # mean = torch.mean(data, dim=(-1, -2), keepdim=True)
        # mean = mean.expand_as(data)
        # data = data - mean

        # Normalization
        std, mean = torch.std_mean(data, dim=(-1, -2), keepdim=True)
        std = std.expand_as(data)
        mean = mean.expand_as(data)
        data = (data - mean) / (std + epsilon)  # size: Sx3xHxW, or S,1,H,W

        result = {"images": data,  # 3,H,W or 1,H,W
                  "GTs": labels,
                  "IDs": ID
                  }
        return result  # B,3,H,W

    def getVolumeItem(self,index):
        startIndex = self.m_volumeStartIndex[index]
        numSlices = self.m_volumeNumSlices[index]

        epsilon = 1e-8
        data_B = []
        labels_B = []
        ID_B = []

        for sliceIndex in range(startIndex, startIndex + numSlices):
            ID = self.m_sliceIDs[sliceIndex]

            labels = []
            if self.hps.existGTLabel:
                labels = self.m_labels[int(ID)]

            slicePath = self.m_slicesPath[sliceIndex]
            npSlice = np.load(slicePath)

            data = torch.from_numpy(npSlice).to(device=self.hps.device, dtype=torch.float32)
            H, W = data.shape

            # transform for data augmentation
            if self.m_transform:
                data = self.m_transform(data)  # size: HxW

            if 0 != self.hps.gradChannels:
                data = self.addSliceGradient(data)  # 3,H,W
            else:
                data = data.unsqueeze(dim=0)  # 1,H,W

            # normalization before output to dataloader
            # AlexNex, GoogleNet V1, VGG, ResNet only do mean subtraction without dividing std.
            # mean = torch.mean(data, dim=(-1, -2), keepdim=True)
            # mean = mean.expand_as(data)
            # data = data - mean

            # Normalization
            std, mean = torch.std_mean(data, dim=(-1, -2), keepdim=True)
            std = std.expand_as(data)
            mean = mean.expand_as(data)
            data = (data - mean) / (std + epsilon)  # size: 3xHxW, or 1,H,W

            data_B.append(data)
            labels_B.append(labels)
            ID_B.append(ID)

        data_B = [item.unsqueeze(dim=0) for item in data_B]
        catData = torch.cat(data_B, dim=0)  # Bx3xHxW

        # concatenate dictionary's value into tensor
        catLabels ={}
        for line in labels_B:
            for key in line:
                if key in catLabels:
                    catLabels[key] += [line[key],]
                else:
                    catLabels[key] = [line[key],]
        for key in catLabels:
            catLabels[key] = torch.tensor(catLabels[key]).to(device=self.hps.device)

        result = {"images": catData,  # B,3,H,W or B,1,H,W
                  "GTs": catLabels,
                  "IDs": ID_B
                  }
        return result  # B,3,H,W, following process needs squeeze its extra batch dimension.









