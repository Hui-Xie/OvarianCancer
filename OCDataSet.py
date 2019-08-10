
#Ovarian Cancer DataSet
import sys
import random
from FilesUtilities import *
import numpy as np
import math
import json

import torch
from torch.utils import data

class OVDataPartition():
    def __init__(self, inputsDir, labelsPath, inputSuffix, K_fold, k, logInfoFun=print):
        self.m_inputsDir = inputsDir
        self.m_inputFilesListFile = os.path.join(self.m_inputsDir, "inputFilesList.txt")
        self.m_labelsPath = labelsPath
        self.m_inputSuffix = inputSuffix
        self.m_KFold = K_fold
        self.m_k = k
        self.m_logInfo = logInfoFun

        self.updateInputsList()
        self.updateLabelsList()

        self.statisticsLabels()
        self.partition()

    def updateInputsList(self):
        self.m_inputFilesList = []
        if os.path.isfile(self.m_inputFilesListFile):
            self.m_inputFilesList = loadInputFilesList(self.m_inputFilesListFile)
        else:
            self.m_inputFilesList = getFilesList(self.m_inputsDir, self.m_inputSuffix)
            self.m_logInfo(
                f"program re-initializes all input files list, which will lead previous all K_fold cross validation invalid.")
            saveInputFilesList(self.m_inputFilesList, self.m_inputFilesListFile)

    def updateLabelsList(self):
        self.m_labelsList = []
        with open(self.m_labelsPath) as f:
            allPatientRespsDict = json.load(f)

        for file in self.m_inputFilesList:
            patientID = getStemName(file, self.m_inputSuffix)
            if len(patientID) > 8:
                patientID = patientID[0:8]
            self.m_labelsList.append(allPatientRespsDict[patientID])

    def statisticsLabels(self):
        self.m_0FileIndices = []
        self.m_1FileIndices = []
        for i, label in enumerate(self.m_labelsList):
            if label ==0:
                self.m_0FileIndices.append(i)
            else:
                self.m_1FileIndices.append(i)
        self.m_logInfo(f"Infor: In all data of {len(self.m_labelsList)} files, label 0 has {len(self.m_0FileIndices)} files,\n\t  and label 1 has {len(self.m_1FileIndices)} files, "\
                       + f"where positive response rate = {len(self.m_1FileIndices)/len(self.m_labelsList)} in full data")

    def partition(self):
        N = len(self.m_labelsList)
        N0 = len(self.m_0FileIndices)
        N1 = len(self.m_1FileIndices)

        random.seed(201908)
        random.shuffle(self.m_0FileIndices)
        random.shuffle(self.m_1FileIndices)

        folds0 = np.array_split(np.asarray(self.m_0FileIndices), self.m_KFold)
        folds1 = np.array_split(np.asarray(self.m_1FileIndices), self.m_KFold)

        self.m_partitions = {}
        k = self.m_k
        K = self.m_KFold
        self.m_partitions["test"] = folds0[k].tolist() + folds1[k].tolist()
        k1 = (k+1)% K  # validation k
        self.m_partitions["validation"] = folds0[k1].tolist() + folds1[k1].tolist()
        self.m_partitions["training"] = []
        for i in range(K):
            if i != k and i != k1:
                self.m_partitions["training"] += folds0[i].tolist() + folds1[i].tolist()
        self.m_logInfo(f"{K}-fold cross validation: the {k}th fold is for test, the {k1}th fold is for validation, remaining folds are for training.")



class OVDataSet(data.Dataset):
    def __init__(self, name, dataPartitions, transform=None, logInfoFun=print):
        self.m_dataPartitions = dataPartitions
        self.m_dataIDs = self.m_dataPartitions.m_partitions[name]
        self.m_transform = transform
        self.m_logInfo = logInfoFun
        self.m_labels = self.getLabels(self.m_dataIDs)
        self.m_logInfo(f" {name} dataset:  total {len(self.m_labels)} files, where 1 has {sum(self.m_labels)} with rate of {sum(self.m_labels) / len(self.m_labels)}")

    def __len__(self):
        return len(self.m_labels)

    def __getitem__(self, index):
        ID = self.m_dataIDs[index]
        filename = self.m_dataPartitions.m_inputFilesList[ID]
        data = np.load(filename).astype('float32')
        label = self.m_labels[index]

        if self.m_transform:
            data = self.m_transform(data)

        return data, label

    def getLabels(self, dataIDs):
        labels = []
        for id in dataIDs:
            label = self.m_dataPartitions.m_labelsList[id]
            labels.append(label)
        return labels