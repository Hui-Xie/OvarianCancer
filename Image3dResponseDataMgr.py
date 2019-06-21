
import os
import numpy as np
import random
import sys
from scipy.misc import imsave
from scipy import ndimage
from ResponseDataMgr import ResponseDataMgr


class Image3dResponseDataMgr(ResponseDataMgr):
    def __init__(self, inputsDir, labelsPath, inputSuffix, K_fold, k, logInfoFun=print):
        super().__init__(inputsDir, labelsPath, inputSuffix, K_fold, k, logInfoFun)

    def dataResponseGenerator(self, inputFileIndices, shuffle=True):
        """
        yield (3DImage  - treatment Response) Tuple

        """
        shuffledList = inputFileIndices.copy()
        if shuffle:
            random.shuffle(shuffledList)

        batch = 0
        dataList=[]  # for yield
        responseList= []

        for i in shuffledList:
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

    def dataSegResponseGenerator(self, inputFileIndices, shuffle=True, convertAllZeroSlices=True):
        """
        yied (3DImage  -- Segmentation --  treatment Response) Tuple

        """
        shuffledList = inputFileIndices.copy()
        if shuffle:
            random.shuffle(shuffledList)

        batch = 0
        dataList = []  # for yield
        segList = []
        responseList = []

        for i in shuffledList:
            imageFile = self.m_inputFilesList[i]

            # for inputSize 147*281*281, and segmentation size of 127*255*255
            # labelFile = imageFile.replace("Images_ROI_29_140_140", "Labels_ROI_23_127_127")
            labelFile = imageFile.replace("images_augmt_29_140_140", "labels_augmt_23_127_127")

            image3d = np.load(imageFile)
            image3d = np.expand_dims(image3d, 0)  # add channel dim as 1
            seg3d   = np.load(labelFile)
            if convertAllZeroSlices:
                self.convertAllZeroSliceToValue(seg3d, -100)  # -100 is default ignore_index in CrossEntropyLoss
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

    def getSegCEWeight(self):
        labelPortion = [0.95995, 0.0254, 0.01462, 0.00003]  # this is portion of 0,1,2,3 label, whose sum = 1
        remainedLabels = (0,1,2)
        N = 3
        ceWeight = [0.0] * N
        accumu = 0.0
        for i, x in enumerate(remainedLabels):
            if 0 == x:
                position0 = i
                continue
            else:
                ceWeight[i] = 1 / labelPortion[x]
                accumu += labelPortion[x]
        ceWeight[position0] = 1 / (1 - accumu)  # unused labels belong to background 0
        self.m_logInfo(f"Infor: Segmentation Cross Entropy Weight: {ceWeight} for label {remainedLabels}")
        return ceWeight


