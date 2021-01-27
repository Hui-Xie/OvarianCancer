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

# use AddSample Augmentation

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

        # load all volumes into memory
        assert hps.imageW == 1
        self.m_volumes = np.empty((self.m_NVolumes, hps.inputChannels, hps.imageH), dtype=np.float)  # size:NxCxH for 9x9 sector array
        for i, volumePath in enumerate(self.m_volumePaths):
            oneVolume = np.load(volumePath).astype(np.float)
            self.m_volumes[i, :] = oneVolume
        self.m_volumes = self.m_volumes.reshape(-1, hps.inputChannels * hps.imageH * hps.imageW)  # size: Nx(CxHxW)

        # read clinical features
        fullLabels = readBESClinicalCsv(hps.GTPath)

        labelTable = np.empty((self.m_NVolumes, 22), dtype=np.float)  # size: Nx22
        # labelTable head: patientID,                                          (0)
        #             "hypertension_bp_plus_history$", "gender", "Age$",'IOP$', 'AxialLength$', 'Height$', 'Weight$', 'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$',
        # columnIndex:         1                           2        3       4          5             6          7             8              9                10
        #              'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',  'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015',
        # columnIndex:   11            12                           13                      14                       15                       16                  17
        #              'TG$_Corrected2015',  BMI,   WaistHipRate,  LDL/HDL
        # columnIndex:      18                 19       20         21
        for i in range(self.m_NVolumes):
            id = int(self.m_IDsCorrespondVolumes[i])
            labelTable[i, 0] = id

            # appKeys: ["hypertension_bp_plus_history$", "gender", "Age$", 'IOP$', 'AxialLength$', 'Height$', 'Weight$',
            #          'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$', 'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',
            #          'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015', 'TG$_Corrected2015']
            for j, key in enumerate(hps.appKeys):
                oneLabel = fullLabels[id][key]
                if "gender" == key:
                    oneLabel = oneLabel - 1
                labelTable[i, 1 + j] = oneLabel

            # compute BMI, WaistHipRate, LDL/HDL
            if labelTable[i, 7] == -100 or labelTable[i, 6] == -100:
                labelTable[i, 19] = -100  # emtpty value
            else:
                labelTable[i, 19] = labelTable[i, 7] / ((labelTable[i, 6] / 100.0) ** 2)  # weight is in kg, height is in cm.

            if labelTable[i, 8] == -100 or labelTable[i, 9] == -100:
                labelTable[i, 20] = -100
            else:
                labelTable[i, 20] = labelTable[i, 8] / labelTable[i, 9]  # both are in cm.

            if labelTable[i, 17] == -100 or labelTable[i, 16] == -100:
                labelTable[i, 21] = -100
            else:
                labelTable[i, 21] = labelTable[i, 17] / labelTable[i, 16]  # LDL/HDL, bigger means more risk to hypertension.

        # concatenate selected thickness and clinical features, and then delete empty-feature patients
        self.m_inputClinicalFeatures = hps.inputClinicalFeatures
        clinicalFeatureColIndex = tuple(hps.clinicalFeatureColIndex)
        nClinicalFtr = len(clinicalFeatureColIndex)
        assert nClinicalFtr == hps.numClinicalFtr

        clinicalFtrs = labelTable[:, clinicalFeatureColIndex]
        # delete the empty value of "-100"
        emptyRows = np.nonzero(clinicalFtrs == -100)
        extraEmptyRows = np.nonzero(clinicalFtrs[:,self.m_inputClinicalFeatures.index("IOP")] == 99)  #missing IOP value
        emptyRows = (np.concatenate((emptyRows[0], extraEmptyRows[0]), axis=0),)

        self.m_inputThicknessFeatures = hps.inputThicknessFeatures
        thicknessFeatureColIndex = tuple(hps.thicknessFeatureColIndex)
        nThicknessFtr = len(thicknessFeatureColIndex)
        assert nThicknessFtr == hps.numThicknessFtr

        thicknessFtrs = self.m_volumes[:,thicknessFeatureColIndex]

        # concatenate sector thickness with multi variables:
        self.m_volumes = np.concatenate((thicknessFtrs, clinicalFtrs), axis=1)  # size: Nx(nThicknessFtr+nClinicalFtr)
        assert self.m_volumes.shape[1] == hps.inputWidth

        self.m_volumes = np.delete(self.m_volumes, emptyRows, 0)
        self.m_targetLabels = np.delete(labelTable, emptyRows, 0)[:,1] # for hypertension

        # convert to torch tensor
        self.m_volumes = torch.from_numpy(self.m_volumes).to(device=hps.device, dtype=torch.float32)
        self.m_targetLabels = torch.from_numpy(self.m_targetLabels).to(device=hps.device, dtype=torch.float32)

        emptyRows = tuple(emptyRows[0])
        self.m_volumePaths = [path for index, path in enumerate(self.m_volumePaths) if index not in emptyRows]
        self.m_IDsCorrespondVolumes = [id for index, id in enumerate(self.m_IDsCorrespondVolumes) if index not in emptyRows]

        # update the number of volumes.
        self.m_NVolumes = len(self.m_volumes)
        assert hps.inputWidth == self.m_volumes.shape[1]

        with open(hps.logMemoPath, "a") as file:
            file.write(f"{mode} data set: NVolumes={self.m_NVolumes}\n")


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

        data = self.m_volumes[index,].clone()  # clone is safer to avoid source data corrupted

        # No transform for data augmentation


        # cancel normalization again, modified at Dec 30th, 2020
        # as input volumes has been normlizated.

        # std, mean = torch.std_mean(data, dim=(1, 2), keepdim=True)
        # std = std.expand_as(data)
        # mean = mean.expand_as(data)
        # data = (data - mean) / (std + epsilon)  # size: CxHxW

        result = {"images": data,
                  "GTs": label,
                  "IDs": volumePath
                  }
        return result









