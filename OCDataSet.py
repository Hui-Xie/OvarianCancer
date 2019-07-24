
#Ovarian Cancer DataSet
import os
from FilesUtilities import *
import numpy as np

import torch
from torch.utils import data

class OVDataPartition(Object):
    def __init__(self, inputsDir, labelsPath, inputSuffix, K_fold, testProportion, logInfoFun=print):
        self.m_inputsDir = inputsDir
        self.m_inputFilesListFile = os.path.join(self.m_inputsDir, "inputFilesList.txt")
        self.m_labelsPath = labelsPath
        self.m_inputSuffix = inputSuffix
        self.m_KFold = K_fold
        self.m_testProportion = testProportion
        self.m_logInfo = logInfoFun

        self.updateInputsList()
        self.updateLabelsList()

        self.statisticsLabels()
        self.partition()

    def updateInputsList(self):
        self.m_inputFilesList = []
        if os.path.isfile(self.m_inputFilesListFile):
            self.m_inputFilesList = loadInputFilesList()
        else:
            self.m_inputFilesList = getFilesList(self.m_inputsDir, self.m_inputSuffix)
            self.m_logInfo(
                f"program re-initializes all input files list, which will lead previous all K_fold cross validation invalid.")
            saveInputFilesList(self.m_inputFilesList, self.m_inputFilesListFile)

    def updateLabelsList(self):
        self.m_lablesList = []
        with open(self.m_labelsPath) as f:
            allPatientRespsDict = json.load(f)

        for file in self.m_inputFilesList:
            patientID = getStemName(file, self.m_inputSuffix)
            if len(patientID) > 8:
                patientID = patientID[0:8]
            self.m_lablesList.append(allPatientRespsDict[patientID])

    def statisticsLabels(self):
        self.m_0FileIndices = []
        self.m_1FileIndices = []
        for i, label in enumerate(self.m_lablesList):
            if label ==0:
                self.m_0FileIndices.append(i)
            else:
                self.m_1FileIndices.append(i)
        self.m_logInfo(f"Infor: In all data of {len(self.m_lablesList)} files, label 0 has {len(self.m_0FileIndices)} files,\n\t  and label 1 has {len(self.m_1FileIndices)} files, "\
                       + f"where positive response rate = {len(self.m_1FileIndices)/len(self.m_labelsList)} in full data")

    def partition(self):
        N = len(self.m_lablesList)
        N0 = len(self.m_0FileIndices)
        N1 = len(self.m_1FileIndices)
        Ntest = int(math.ceil(N*self.m_testProportion))

        random.seed(201907)
        random.shuffle(self.m_0FileIndices)
        random.shuffle(self.m_1FileIndices)

        nTest0 = int(N0 * self.m_testProportion)
        nTest1 = Ntest- nTest0

        self.m_partition = {}
        self.m_partition["test"]  = self.m_0FileIndices[0:nTest0] + self.m_1FileIndices[0:nTest1]
        self.m_partition["train0s"] =  np.asarray(self.m_0FileIndices[nTest0:]).split(self.m_KFold).tolist()
        self.m_partition["train1s"] =  np.asarray(self.m_1FileIndices[nTest1:]).split(self.m_KFold).tolist()

        self.m_logInfo(f"Infor: independent test Set has {N} files,and Training Set has {N-Ntest} files which will be divided into {self.m_KFold} folds.")

    def getLabels(self, dataIDs):
        labels =[]
        for id in dataIDs:
            label = self.m_lablesList[id]
            labels.append(label)
        return labels

class OVDataSet(data.DataSet):
    def __init__(self, dataPartitions, partitionName, k, logInfoFun=print):
        self.m_dataPartioins = dataPartitions
        if "test" == partitionName:
            self.m_dataIDs = self.m_dataPartioins.m_partition["test"]
            self.m_labels  = self.m_dataPartioins.getLabels(self.m_dataIDs)
        elif "train" == partitionName and k < self.m_dataPartioins.m_KFold:

        else:
            self.m_logInfo("Error: partitionName in wrong")
            sys.exit(0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y
