from torch.utils import data
import numpy as np
import torch
import os
import random

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

        The inputData has done below:
        A. Input: 31x496x512 in pixel space, where 31 is the number of B-scans.
        B. Use segmented mask to exclude non-retina region, e.g. vitreous body and choroid etc, getting 31x496x512 filtered volume.
        C. use (x-mu)/sigma to normalization in the whole data along each B-scan.

        This dataSet need to do below:
        A. Random crop the filtered 3D volume into 31x448x448 as data augmentation.
        D. halve size in X and Y direction to scale down to 31x224x224 to feed into network.

        '''
        super().__init__()
        self.m_mode = mode
        self.hps = hps

        if mode == "training":
            p = hps.trainingDataPath
        elif mode == "validation":
            p = hps.validationDataPath
        elif mode == "test":
            p = hps.testDataPath
        else:
            print(f"OCT2SysDiseaseDataSet mode error")
            assert False
        IDPath = p[:p.rfind(".csv")] + f"{hps.k}.csv"  # add k to IDPath

        # read GT
        self.m_labels = readBESClinicalCsv(hps.GTPath)

        with open(IDPath, 'r') as idFile:
            IDList = idFile.readlines()
        IDList = [item[0:-1] for item in IDList]  # erase '\n'

        self.m_transform = transform

        # get all correct volume numpy path
        allVolumesList = glob.glob(hps.dataDir + f"/*{hps.volumeSuffix}")
        incorrectIDList = []

        # make sure volume ID and volume path has strict corresponding order
        self.m_volumePaths = []  # number of volumes is exact 2 times of IDList for ODOS matches
        self.m_IDsCorrespondVolumes = []

        volumePathsFile = os.path.join(hps.dataDir, self.m_mode+f"_{hps.ODOS}_VolumePaths_{hps.K_fold}CV_{hps.k}.txt")
        IDsCorrespondVolumesPathFile = os.path.join(hps.dataDir, self.m_mode+f"_{hps.ODOS}_IDsCorrespondVolumes_{hps.K_fold}CV_{hps.k}.txt")

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
                # for OD and OS
                ODOSResultList = fnmatch.filter(allVolumesList, "*/" + ID + f"_O[D,S]_*{hps.volumeSuffix}")
                if len(ODOSResultList) > 0 :
                    self.m_volumePaths += ODOSResultList
                    self.m_IDsCorrespondVolumes += [ID,]*len(ODOSResultList)
                else:
                    incorrectIDList.append(ID)

            if len(incorrectIDList) > 0:
                with open(hps.logMemoPath, "a") as file:
                    file.write(f"incorrect ID List of {hps.ODOS} in {mode}_{hps.K_fold}CV_{hps.k} (missing):\n {incorrectIDList}")

            # save files
            with open(volumePathsFile, "w") as file:
                for v in self.m_volumePaths:
                    file.write(f"{v}\n")
            with open(IDsCorrespondVolumesPathFile, "w") as file:
                for v in self.m_IDsCorrespondVolumes:
                    file.write(f"{v}\n")

        self.m_NVolumes = len(self.m_IDsCorrespondVolumes)  # number of OD and OS volumes
        assert self.m_NVolumes  == len(self.m_volumePaths)

        # 3D volume is too big, it can not load all volumes into memory

        # read clinical features
        fullLabels = self.m_labels

        # get HBP labels
        self.m_labelTable = np.empty((self.m_NVolumes, 2), dtype=np.int)  # size: Nx2 with (ID, Hypertension)
        for i in range(self.m_NVolumes):
            id = int(self.m_IDsCorrespondVolumes[i])
            self.m_labelTable[i, 0] = id
            self.m_labelTable[i, 1] = fullLabels[id][hps.targetKey]
        self.m_targetLabels = self.m_labelTable[:, 1]  # for hypertension
        # convert to torch tensor
        self.m_targetLabels = torch.from_numpy(self.m_targetLabels).to(device=hps.device, dtype=torch.float32)

        self.m_rH = hps.originalImageH - 2 * hps.imageH  #  randomRangeH
        self.m_rW = hps.originalImageW - 2 * hps.imageW  # randomRangeW

        # save log information:
        with open(hps.logMemoPath, "a") as file:
            file.write(f"{mode} dataset in {hps.K_fold}CV_{hps.k}: NVolumes={self.m_NVolumes}\n")
            rate1 = self.m_labelTable[:, 1].sum() * 1.0 / self.m_NVolumes
            rate0 = 1 - rate1
            file.write(f"{mode} dataset in {hps.K_fold}CV_{hps.k}: proportion of 0,1 = [{rate0},{rate1}]\n")

    def __len__(self):
        return self.m_NVolumes

    def getGTDict(self):
        return self.m_labels

    def addSamplesAugmentation(self, data0, label0):
        index1, index2 = random.sample(range(self.m_NVolumes), 2)
        volumePath = "__AddSamples_MergerPath__"

        label1 = self.m_labels[int(self.m_IDsCorrespondVolumes[index1])][self.hps.appKey]
        if "gender" == self.hps.appKey:
            label1 = label1 - 1
        label2 = self.m_labels[int(self.m_IDsCorrespondVolumes[index2])][self.hps.appKey]
        if "gender" == self.hps.appKey:
            label2 = label2 - 1

        if label0 == label1 == label2:
            data1 = self.m_volumes[index1,].clone()  # clone is safer to avoid source data corrupted
            data2 = self.m_volumes[index2,].clone()  # clone is safer to avoid source data corrupted
            data = (data0+ data1+data2)/3.0
            label = label0

        elif label0 == label1:
            data1 = self.m_volumes[index1,].clone()  # clone is safer to avoid source data corrupted
            data = (data0 + data1) / 2.0
            label = label0
        elif label0 == label2:
            data2 = self.m_volumes[index2,].clone()  # clone is safer to avoid source data corrupted
            data = (data0 + data2) / 2.0
            label = label0
        elif label1 == label2:
            data1 = self.m_volumes[index1,].clone()  # clone is safer to avoid source data corrupted
            data2 = self.m_volumes[index2,].clone()  # clone is safer to avoid source data corrupted
            data = (data1 + data2) / 2.0
            label = label1
        else:
            print(f"3 samples have 3 different labels in binary labels case")
            assert False

        return data, label, volumePath


    def __getitem__(self, index):
        epsilon = 1.0e-8
        volumePath = self.m_volumePaths[index]

        label = None
        if self.hps.existGTLabel:
            label = self.m_targetLabels[index]

        data = np.load(self.m_volumePaths[index]).astype(float)
        data =  torch.from_numpy(data).to(device=self.hps.device, dtype=torch.float32)

        H2 = self.hps.imageH * 2
        W2 = self.hps.imageW * 2

        if self.m_mode == "test":
            # 5-crop images and mirror -> 10 crops.
            '''
            # Test-time augmentation:
            # refer to AlexNet's idea,and ResNet also adapted this 10-crop TTA:
            #  "At test time, the network makes a prediction by extracting five 224×224 patches
            #  (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all),
            #  and averaging the predictions made by the network’s softmax layer on the ten patches."
            
            crop locations: 
            0   1
              4
            2   3
            '''

            data10Crops = torch.zeros((10,self.hps.inputChannels, self.hps.imageH, self.hps.imageW), dtype=torch.float32, device=self.hps.device)
            crop0 = data[:, 0:H2, 0:W2][:,0::2, 0::2]
            crop1 = data[:, 0:H2,  self.m_rW:self.m_rW + W2][:, 0::2, 0::2]
            crop2 = data[:, self.m_rH:self.m_rH + H2,  0:W2][:, 0::2, 0::2]
            crop3 = data[:, self.m_rH:self.m_rH + H2,  self.m_rW:self.m_rW + W2][:, 0::2, 0::2]
            crop4 = data[:, self.m_rH // 2:self.m_rH // 2 + H2,  self.m_rW // 2:self.m_rW // 2 + W2][:, 0::2, 0::2]
            data10Crops[0,] = crop0.clone()
            data10Crops[1,] = crop1.clone()
            data10Crops[2,] = crop2.clone()
            data10Crops[3,] = crop3.clone()
            data10Crops[4,] = crop4.clone()
            data10Crops[5,] = crop0.flip(dims=(2,))
            data10Crops[6,] = crop1.flip(dims=(2,))
            data10Crops[7,] = crop2.flip(dims=(2,))
            data10Crops[8,] = crop3.flip(dims=(2,))
            data10Crops[9,] = crop4.flip(dims=(2,))
            data = data10Crops

        else: # for validation and training data
            h = np.random.randint(self.m_rH)
            w = np.random.randint(self.m_rW)
            data = data[:,h:h+H2,w:w+W2]  # crop to H2xW2
            data = data[:,0::2, 0::2]   # halve the size in H, and W

        if self.m_transform:
            data = self.m_transform(data)

        result = {"images": data,
                  "GTs": label,
                  "IDs": volumePath
                  }
        return result









