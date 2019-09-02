
#Ovarian Cancer Data and Segmentation dataset

import sys
import random
from FilesUtilities import *
import numpy as np
import math
import json
import torchvision.transforms.functional as TF

import torch
from torch.utils import data

class OVDataSegPartition():
    def __init__(self, inputDataDir, inputLabelDir=None, inputSuffix="", K_fold=5, k=0, logInfoFun=print):
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

        random.seed(201908)
        random.shuffle(self.m_filesIndex)

        folds = np.array_split(np.asarray(self.m_filesIndex), self.m_KFold)

        self.m_partitions = {}
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
        data = np.load(filename).astype(np.float32)

        patientID = getStemName(filename, self.m_dataPartitions.m_inputSuffix)
        if self.m_dataPartitions.m_inputLabelDir is not None:
            labelFile = os.path.join(self.m_dataPartitions.m_inputLabelDir, patientID+self.m_dataPartitions.m_inputSuffix)
            label = np.load(labelFile).astype(np.float32)

            if self.m_transform:
                data, label = self.m_transform(data, label)
            else:
                data, label = torch.from_numpy(data), torch.from_numpy(label)

            return data, label
        else:
            return torch.from_numpy(data), patientID