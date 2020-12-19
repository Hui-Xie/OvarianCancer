from torch.utils import data
import numpy as np
import torch
import os

import glob
import fnmatch
import yaml

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
        allVolumesList = glob.glob(hps.dataDir + f"/*{hps.volumeSuffix}")
        nonexistIDList = []

        # make sure volume ID and volume path has strict corresponding order
        self.m_volumePaths = []  # number of volumes is about 2 times of IDList
        self.m_IDsCorrespondVolumes = []

        volumePathsFile = os.path.join(hps.dataDir, self.m_mode+"_VolumePaths.txt")
        IDsCorrespondVolumesPathFile = os.path.join(hps.dataDir, self.m_mode+"_IDsCorrespondVolumes.txt")

        # save related file in order to speed up.
        if os.path.isfile(volumePathsFile) and os.path.isfile(IDsCorrespondVolumesPathFile):
            with open(volumePathsFile, 'r') as file:
                lines = file.readlines()
            self.m_volumePaths = [item[0:-1] for item in lines]  # erase '\n'

            with open(IDsCorrespondVolumesPathFile, 'r') as file:
                lines = file.readlines()
            self.m_IDsCorrespondVolumes = [item[0:-1] for item in lines]  # erase '\n'

        else:
            for i,ID in enumerate(IDList):
                resultList = fnmatch.filter(allVolumesList, "*/" + ID + f"_O[D,S]_*{hps.volumeSuffix}")
                resultList.sort()
                numVolumes = len(resultList)
                if 0 == numVolumes:
                    nonexistIDList.append(ID)
                else:
                    self.m_volumePaths += resultList
                    self.m_IDsCorrespondVolumes += [ID,]*numVolumes

            if len(nonexistIDList) > 0:
                print(f"Error: nonexistIDList:\n {nonexistIDList}")
                assert False

            # save files
            with open(volumePathsFile, "w") as file:
                for v in self.m_volumePaths:
                    file.write(f"{v}\n")
            with open(IDsCorrespondVolumesPathFile, "w") as file:
                for v in self.m_IDsCorrespondVolumes:
                    file.write(f"{v}\n")

        self.m_NVolumes = len(self.m_volumePaths)
        assert (self.m_NVolumes == len(self.m_IDsCorrespondVolumes))

        #load all volumes into memory, which needs about 2.6GB memory in BES_3K thickness data.
        self.m_volumes = torch.empty((self.m_NVolumes, hps.inputChannels, hps.imageH, hps.imageW), device=hps.device, dtype=torch.float)
        for i, volumePath in enumerate(self.m_volumePaths):
            oneVolume = torch.from_numpy(np.load(volumePath)).to(device=hps.device, dtype=torch.float32)
            self.m_volumes[i,] = oneVolume

        # normalize training volumes and save mean and std for using in validation and test data
        # normalization along each C (layer) dimension, and save as torch.pt file.
        epsilon = 1.0e-8
        self.m_normalizationFilePath_std = os.path.join(hps.netPath, hps.trainNormalizationStdMeanFileName +"_std.pt")
        self.m_normalizationFilePath_mean = os.path.join(hps.netPath, hps.trainNormalizationStdMeanFileName + "_mean.pt")
        if mode == "training":
            std, mean = torch.std_mean(self.m_volumes, dim=(0,2,3),keepdim=True)
            torch.save(std, self.m_normalizationFilePath_std)
            torch.save(mean, self.m_normalizationFilePath_mean)
            std = std.expand_as(self.m_volumes)
            mean = mean.expand_as(self.m_volumes)
            self.m_volumes = (self.m_volumes - mean) / (std + epsilon)  # size: NxCxHxW

        elif (mode == "validation") or (mode == "test"):
            std = torch.load(self.m_normalizationFilePath_std).to(device=hps.device).expand_as(self.m_volumes)
            mean = torch.load(self.m_normalizationFilePath_mean).to(device=hps.device).expand_as(self.m_volumes)
            self.m_volumes = (self.m_volumes - mean) / (std + epsilon)  # size: NxCxHxW
        else:
            print(f"OCT2SysDiseaseDataSet mode error")
            assert False



    def __len__(self):
        return self.m_NVolumes

    def getGTDict(self):
        return self.m_labels



    def __getitem__(self, index):
        epsilon = 1.0e-8
        volumePath = self.m_volumePaths[index]

        label = None
        if self.hps.existGTLabel:
            ID = self.m_IDsCorrespondVolumes[index]
            label = self.m_labels[int(ID)][self.hps.appKey]
            if "gender" == self.hps.appKey:
                label = label - 1

        data = self.m_volumes[index,].clone()  # clone is safer to avoid source data corrupted
        C, H, W = data.shape
        if (H != self.hps.imageH) or (W != self.hps.imageW) or (C != self.hps.inputChannels):
            print(f"Error: {volumePath} has incorrect size C= {C}, H={H} and W={W}, ")
            assert False

        # transform for data augmentation
        if self.m_transform:
            data = self.m_transform(data)  # size: CxHxW

        # texture map needs normalization again.
        std, mean = torch.std_mean(data, dim=(1,2), keepdim=True)
        std = std.expand_as(data)
        mean = mean.expand_as(data)
        data = (data - mean) / (std + epsilon)  # size: CxHxW

        result = {"images": data,  # B,C,H,W
                  "GTs": label,
                  "IDs": volumePath
                  }
        return result  # B,3,H,W, following process needs squeeze its extra batch dimension.









