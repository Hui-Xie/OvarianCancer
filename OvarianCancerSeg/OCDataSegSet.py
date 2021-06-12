
#Ovarian Cancer Data and Segmentation dataset

import sys
import random
from utilities.FilesUtilities import *
import numpy as np
import math
import json
import torchvision.transforms.functional as TF

import torch
from torch.utils import data

class OVDataSegPartition():
    def __init__(self, inputDataDir, inputLabelDir=None, inputSuffix="", K_fold=0, k=0, logInfoFun=print):
        self.m_inputDataDir = inputDataDir
        self.m_inputDataListFile = os.path.join(self.m_inputDataDir, "inputFilesList.txt")
        self.m_inputLabelDir = inputLabelDir
        self.m_inputSuffix = inputSuffix
        self.m_KFold = K_fold
        self.m_k = k
        self.m_logInfo = logInfoFun

        self.updateInputsList()
        self.partition()

    def updateInputsList(self):
        self.m_inputFilesList = []
        if os.path.isfile(self.m_inputDataListFile):
            self.m_inputFilesList = loadInputFilesList(self.m_inputDataListFile)
        else:
            self.m_inputFilesList = getFilesList(self.m_inputDataDir, self.m_inputSuffix)
            self.m_logInfo(
                f"program re-initializes all input files list, which will lead previous all K_fold cross validation invalid.")
            saveInputFilesList(self.m_inputFilesList, self.m_inputDataListFile)

    def partition(self):
        N = len(self.m_inputFilesList)
        self.m_filesIndex = list(range(N))
        self.m_partitions = {}

        if 0 == self.m_KFold or 1 == self.m_KFold:
            self.m_partitions["all"] = self.m_filesIndex
            self.m_logInfo(f"All files are in one partition.")
        else:
            random.seed(201908)
            random.shuffle(self.m_filesIndex)

            folds = np.array_split(np.asarray(self.m_filesIndex), self.m_KFold)

            k = self.m_k
            K = self.m_KFold
            self.m_partitions["fulldata"] = self.m_filesIndex
            self.m_partitions["test"] = folds[k].tolist()
            k1 = (k+1)% K  # validation k
            self.m_partitions["validation"] = folds[k1].tolist()
            self.m_partitions["training"] = []
            for i in range(K):
                if i != k and i != k1:
                    self.m_partitions["training"] += folds[i].tolist()
            if self.m_inputLabelDir is  not None:
                self.m_logInfo(f"{K}-fold cross validation: the {k}th fold is for test, the {k1}th fold is for validation, remaining folds are for training.")

    def getLossWeight(self):
        count1 = 0
        countAll = 0

        if "all" in self.m_partitions.keys():
            trainingPartition = self.m_partitions["all"]
        elif "training" in  self.m_partitions.keys():
            trainingPartition = self.m_partitions["training"]
        else:
            print("Error: program can not find all or training partition in OVDataSegPartition")
            exit(-1)

        countFiles = len(trainingPartition)
        for index in trainingPartition:
            filename = self.m_inputFilesList[index]
            patientID = getStemName(filename, self.m_inputSuffix)
            labelFile = os.path.join(self.m_inputLabelDir,patientID + self.m_inputSuffix)
            label = np.load(labelFile)
            count1 += np.sum((label > 0).astype(int))
            countAll += label.size
        count0 = countAll - count1
        self.m_logInfo(f"Total {countFiles} training files  extracted from {self.m_inputLabelDir}")
        self.m_logInfo(f"0 has {count0} elements, with a rate of  {count0 / countAll} ")
        self.m_logInfo(f"1 has {count1} elements, with a rate of  {count1 / countAll} ")
        lossWeight = torch.tensor([1.0, count0 / count1], dtype=torch.float)
        self.m_logInfo(f"loss weight = {lossWeight}")
        return lossWeight

class OVDataSegSet(data.Dataset):
    def __init__(self, name, dataPartitions, transform=None, logInfoFun=print):
        self.m_dataPartitions = dataPartitions
        self.m_dataIDs = self.m_dataPartitions.m_partitions[name]
        self.m_transform = transform
        self.m_logInfo = logInfoFun
        self.m_logInfo(f"\n{name} dataset: total {len(self.m_dataIDs)} image files.")

    def __len__(self):
        return len(self.m_dataIDs)

    def __getitem__(self, index):
        ID = self.m_dataIDs[index]
        filename = self.m_dataPartitions.m_inputFilesList[ID]
        data = np.load(filename).astype(float)

        patientID = getStemName(filename, self.m_dataPartitions.m_inputSuffix)
        if self.m_dataPartitions.m_inputLabelDir is not None:
            labelFile = os.path.join(self.m_dataPartitions.m_inputLabelDir, patientID+self.m_dataPartitions.m_inputSuffix)
            label = np.load(labelFile).astype(float)

            if self.m_transform:
                data, label = self.m_transform(data, label)
            else:
                data, label = torch.from_numpy(data), torch.from_numpy(label)

            return data.unsqueeze(dim=0), label, patientID  # 3D data filter needs unsueeze feature dim
        else:
            return torch.from_numpy(data), patientID