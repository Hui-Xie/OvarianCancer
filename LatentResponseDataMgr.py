
import os
import numpy as np
import random
import sys
from scipy.misc import imsave
from ResponseDataMgr import ResponseDataMgr
import json

class LatentResponseDataMgr(ResponseDataMgr):
    def __init__(self, inputsDir, labelsPath, logInfoFun=print):
        super().__init__(inputsDir, labelsPath, logInfoFun)
        self.m_inputsSuffix = "_Latent.npy"
        self.initializeInputsResponseList()

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
                batch = 0
                dataList.clear()
                labelList.clear()
                if self.m_oneSampleTraining:
                    break

            latentFile = self.m_inputFilesList[n]
            label = self.m_labelsList[n]
            latent = np.load(latentFile)

            dataList.append(latent)
            labelList.append(label)
            batch +=1

        #  a batch size of 1 and a single feature per channel will has problem in batchnorm.
        #  drop_last data.
        #if 0 != len(dataList) and 0 != len(labelList): # PyTorch supports dynamic batchSize.
        #    yield np.stack(dataList, axis=0), np.stack(labelList, axis=0)

        # clean field
        dataList.clear()
        labelList.clear()


