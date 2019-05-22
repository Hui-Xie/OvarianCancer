
import os
import numpy as np
import random
import sys
from scipy.misc import imsave
from DataMgr import DataMgr
import json

class LatentDataMgr(DataMgr):
    def __init__(self, imagesDir, labelsPath, logInfoFun=print):
        super.__init__(imagesDir, labelsPath, logInfoFun)
        self.m_inputFilesList = self.getFilesList(self.m_inputsDir, "_Latent.npy")
        self.m_labelsList = []
        self.getLabelsList()

    def getLabelsList(self):
        with open(self.m_labelsDir) as f:
            allPatientRespsDict = json.load(f)

        for latent in self.m_inputFilesList:
            patientID = self.getStemName(latent, "_Latent.npy")
            if len(patientID)>8:
                patientID = patientID[0:8]
            self.m_labelsList.append(allPatientRespsDict[patientID])

    def dataLabelGenerator(self, shuffle):
        """
        for simple sample input to the predict network: the input size: 1536*51*49,
             where 1536 = 16*96, number of feature map
                    51   : number of layers
                    49   : feature plane flatted in one dimension.

        """
        self.m_shuffle = shuffle
        N = len(self.m_inputFilesList)
        shuffleList = list(range(N))
        if self.m_shuffle:
            random.shuffle(shuffleList)

        batch = 0
        dataList=[]  # for yield
        labelList= []

        for n in shuffleList:
            if batch >= self.m_batchSize:
                yield np.stack(dataList, axis=0), np.stack(labelList, axis=0)
                if self.m_oneSampleTraining:
                    continue
                else:
                    batch = 0
                    dataList.clear()
                    labelList.clear()

            latentFile = self.m_inputFilesList[n]
            label = self.m_labelsList[n]
            latent = np.load(latentFile)

            dataList.append(latent)
            labelList.append(label)
            batch +=1

        if 0 != len(dataList) and 0 != len(labelList): # PyTorch supports dynamic batchSize.
            yield np.stack(dataList, axis=0), np.stack(labelList, axis=0)

        # clean field
        dataList.clear()
        labelList.clear()


    def getCEWeight(self):  # Todo modify basiing on data statistics of classes in training sample.
        labelPortion = [0.6, 0.4]  # this is portion of 0,1 label, whose sum = 1
        N = len(self.m_remainedLabels)
        ceWeight = [0.0] * N
        accumu = 0.0
        for i, x in enumerate(self.m_remainedLabels):
            if 0 == x:
                position0 = i
                continue
            else:
                ceWeight[i] = 1 / labelPortion[x]
                accumu += labelPortion[x]
        ceWeight[position0] = 1 / (1 - accumu)  # unused labels belong to background 0
        self.m_logInfo(f"Infor: Cross Entropy Weight: {ceWeight}")
        return ceWeight
