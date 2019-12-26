
import os
import numpy as np
import random
import sys
from scipy.misc import imsave
from latentVector.ResponseDataMgr import ResponseDataMgr
import json

class LatentResponseDataMgr(ResponseDataMgr):
    def __init__(self, inputsDir, labelsPath, inputSuffix,  K_fold, k, logInfoFun=print):
        super().__init__(inputsDir, labelsPath, inputSuffix,  K_fold, k, logInfoFun)


    def dataResponseGenerator(self, inputFileIndices, shuffle=True):
        """
        for simple sample input to the predict network: the input size: 1536*51*49,
             where 1536 = 16*96, number of feature map
                    51   : number of layers
                    49   : feature plane flatted in one dimension.

        """
        shuffledList = inputFileIndices.copy()
        if shuffle:
            random.shuffle(shuffledList)

        batch = 0
        dataList=[]  # for yield
        responseList= []

        for i in shuffledList:
            latentFile = self.m_inputFilesList[i]
            label = self.m_responseList[i]
            latent = np.load(latentFile)

            dataList.append(latent)
            responseList.append(label)
            batch +=1

            if batch >= self.m_batchSize:
                yield np.stack(dataList, axis=0), np.stack(responseList, axis=0)
                batch = 0
                dataList.clear()
                responseList.clear()
                if self.m_oneSampleTraining:
                    break

        #  a batch size of 1 and a single feature per channel will has problem in batchnorm.
        #  drop_last data.
        #if 0 != len(dataList) and 0 != len(responseList): # PyTorch supports dynamic batchSize.
        #    yield np.stack(dataList, axis=0), np.stack(responseList, axis=0)

        # clean field
        dataList.clear()
        responseList.clear()


