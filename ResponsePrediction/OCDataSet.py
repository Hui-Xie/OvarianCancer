
#Ovarian Cancer DataSet
import sys
import random
from utilities.FilesUtilities import *
import numpy as np
import math
import json

import torch
from torch.utils import data

class OVDataPartition():
    """
    here: labelsPath is the response file path
    """
    def __init__(self, inputsDir, labelsPath, inputSuffix, K_folds=0, k=0, logInfoFun=print):
        self.m_inputsDir = inputsDir
        self.m_inputFilesListFile = os.path.join(self.m_inputsDir, "inputFilesList.txt")
        self.m_labelsPath = labelsPath
        self.m_inputSuffix = inputSuffix
        self.m_KFold = K_folds
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
            resultsDict = json.load(f)

        for file in self.m_inputFilesList:
            patientID = getStemName(file, self.m_inputSuffix)
            if len(patientID) > 8:
                patientID = patientID[0:8]
            self.m_labelsList.append(resultsDict[patientID])

    def statisticsLabels(self):
        self.m_0FileIndices = []
        self.m_1FileIndices = []
        for i, label in enumerate(self.m_labelsList):
            if isinstance(label, list):
                label = label[0]
            if label ==0:
                self.m_0FileIndices.append(i)
            else:
                self.m_1FileIndices.append(i)

        if isinstance(self.m_labelsList[0], list):
            self.m_logInfo("Program statistics label according to label[0] for each sample.")
        self.m_logInfo(f"Infor: In all data of {len(self.m_labelsList)} files, label 0 has {len(self.m_0FileIndices)} files,\n\t  and label 1 has {len(self.m_1FileIndices)} files, " \
                + f"where positive response rate = {len(self.m_1FileIndices) / len(self.m_labelsList)} in full data")

    def partition(self):
        self.m_partitions = {}

        if 0==self.m_KFold or 1==self.m_KFold:
            self.m_partitions["all"] = self.m_0FileIndices + self.m_1FileIndices
            self.m_logInfo(f"All files are in one partition.")
        else:
            random.seed(201908)
            random.shuffle(self.m_0FileIndices)
            random.shuffle(self.m_1FileIndices)

            folds0 = np.array_split(np.asarray(self.m_0FileIndices), self.m_KFold)
            folds1 = np.array_split(np.asarray(self.m_1FileIndices), self.m_KFold)

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

    def getPositiveWeight(self):
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
            label = self.m_labelsList[index]
            count1 += label
            countAll += 1
        count0 = countAll - count1
        self.m_logInfo(f"In this training partition:")
        self.m_logInfo(f"0 has {count0:.0f} files, with a rate of  {count0 / countAll} ")
        self.m_logInfo(f"1 has {count1:.0f} files, with a rate of  {count1 / countAll} ")
        posWeight = torch.tensor([count0*1.0 / count1], dtype=torch.float)
        self.m_logInfo(f"Positive weight = {posWeight}")
        return posWeight


class OVDataSet(data.Dataset):
    def __init__(self, name, dataPartitions, transform=None, logInfoFun=print, preLoadData=False):
        self.m_dataPartitions = dataPartitions
        self.m_dataIDs = self.m_dataPartitions.m_partitions[name]
        self.m_transform = transform
        self.m_logInfo = logInfoFun
        self.m_labels = self.getLabels(self.m_dataIDs)

        self.m_preLoadData = preLoadData
        if self.m_preLoadData:
           for i, dataID in enumerate(self.m_dataIDs):
               filename = self.m_dataPartitions.m_inputFilesList[dataID]
               data = np.load(filename).astype(float)
               self.m_loadData = np.concatenate((self.m_loadData, data)) if i!=0 else data
           self.m_loadData = self.m_loadData.reshape((i+1,-1))

        if isinstance(self.m_labels[0], float):
            self.m_logInfo(f"{name} dataset:\t total {len(self.m_labels)} files, where 1 has {sum(self.m_labels):.0f} with rate of {sum(self.m_labels) / len(self.m_labels)}")
        elif isinstance(self.m_labels[0], list):
            labelsArray = np.array(self.m_labels)
            shape = labelsArray.shape
            self.m_logInfo(f"\n\n{name} dataset:\t total {shape} ground truth\n")
            for i in range(shape[1]):
                self.m_logInfo(f" \t\t in the {i}-th coloumn, 1 has {np.sum(labelsArray[:,i])} with rate of {np.sum(labelsArray[:,i]) / shape[0]}")
        else:
            self.m_logInfo(f"something wrong in OVDataSet init function")

    def __len__(self):
        return len(self.m_labels)

    def __getitem__(self, index):
        ID = self.m_dataIDs[index]
        filename = self.m_dataPartitions.m_inputFilesList[ID]
        patientID = getStemName(filename, self.m_dataPartitions.m_inputSuffix)
        if self.m_preLoadData:
            data = self.m_loadData[index,]
        else:
            data = np.load(filename).astype(float)
        label = self.m_labels[index]
        if isinstance(label, list):
            label = torch.tensor(label).t()

        if self.m_transform:
            data = self.m_transform(data)
        else:
            data = torch.from_numpy(data)
        if data.squeeze().dim() ==1:
            return  data, label, patientID
        else:
            return data.unsqueeze(dim=0), label, patientID

    def getLabels(self, dataIDs):
        labels = []
        for id in dataIDs:
            label = self.m_dataPartitions.m_labelsList[id]
            labels.append(label)
        return labels