
import os
import numpy as np
import random
import sys
from scipy.misc import imsave
from scipy import ndimage
from ResponseDataMgr import ResponseDataMgr


class Image3dResponseDataMgr(ResponseDataMgr):
    def __init__(self, inputsDir, labelsPath, inputSuffix, logInfoFun=print):
        super().__init__(inputsDir, labelsPath, inputSuffix, logInfoFun)

    def dataResponseGenerator(self, shuffle):
        """
        yield (3DImage  - treatment Response) Tuple

        """
        self.m_shuffle = shuffle
        N = len(self.m_inputFilesList)
        shuffleList = list(range(N))
        if self.m_shuffle:
            random.shuffle(shuffleList)

        batch = 0
        dataList=[]  # for yield
        responseList= []

        for i in shuffleList:
            imageFile = self.m_inputFilesList[i]
            if "_CT.nrrd" == self.m_inputSuffix:
                image3d = self.readImageFile(imageFile)
                shape = image3d.shape
                zoomFactor = [self.m_depth / shape[0], self.m_height / shape[1], self.m_width / shape[2]]
                image3d = ndimage.zoom(image3d, zoomFactor)
            else:  # load zoomed and fixed-size numpy array
                image3d = np.load(imageFile)

            image3d = np.expand_dims(image3d, 0)  # add channel dim as 1
            response = self.m_responseList[i]

            dataList.append(image3d)
            responseList.append(response)
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

    def dataSegResponseGenerator(self, shuffle):
        """
        yied (3DImage  -- Segmentation --  treatment Response) Tuple

        """
        self.m_shuffle = shuffle
        N = len(self.m_inputFilesList)
        shuffleList = list(range(N))
        if self.m_shuffle:
            random.shuffle(shuffleList)

        batch = 0
        dataList = []  # for yield
        segList = []
        responseList = []

        for i in shuffleList:
            imageFile = self.m_inputFilesList[i]
            labelFile = imageFile.replace("Images_", "Labels_")

            image3d = np.load(imageFile)
            image3d = np.expand_dims(image3d, 0)  # add channel dim as 1
            seg3d   = np.load(labelFile)
            seg3d   = np.expand_dims(seg3d, 0)   # add channel dim as 1

            response = self.m_responseList[i]

            dataList.append(image3d)
            segList.append(seg3d)
            responseList.append(response)
            batch += 1

            if batch >= self.m_batchSize:
                yield np.stack(dataList, axis=0), np.stack(segList, axis=0), np.stack(responseList, axis=0)
                batch = 0
                dataList.clear()
                segList.clear()
                responseList.clear()
                if self.m_oneSampleTraining:
                    break

        #  a batch size of 1 and a single feature per channel will has problem in batchnorm.
        #  drop_last data.
        # if 0 != len(dataList) and 0 != len(responseList): # PyTorch supports dynamic batchSize.
        #    yield np.stack(dataList, axis=0), np.stack(responseList, axis=0)

        # clean field
        dataList.clear()
        responseList.clear()




