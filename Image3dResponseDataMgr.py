
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

    def dataLabelGenerator(self, shuffle):
        """
        3D - treatResponse pair

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

            imageFile = self.m_inputFilesList[n]
            if "_CT.nrrd" == self.m_inputSuffix:
                image3d = self.readImageFile(imageFile)
                shape = image3d.shape
                zoomFactor = [self.m_depth / shape[0], self.m_height / shape[1], self.m_width / shape[2]]
                image3d = ndimage.zoom(image3d, zoomFactor)
            else:  # load zoomed and fixed-size numpy array
                image3d = np.load(imageFile)

            image3d = np.expand_dims(image3d, 0)  # add channel dim as 1
            label = self.m_labelsList[n]

            dataList.append(image3d)
            labelList.append(label)
            batch +=1

        #  a batch size of 1 and a single feature per channel will has problem in batchnorm.
        #  drop_last data.
        #if 0 != len(dataList) and 0 != len(labelList): # PyTorch supports dynamic batchSize.
        #    yield np.stack(dataList, axis=0), np.stack(labelList, axis=0)

        # clean field
        dataList.clear()
        labelList.clear()


