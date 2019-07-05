
import os
import numpy as np
import random
import json
import sys
from scipy.misc import imsave
from scipy import ndimage
from ResponseDataMgr import ResponseDataMgr


class Image3dResponseDataMgr(ResponseDataMgr):
    def __init__(self, inputsDir, labelsPath, inputSuffix, K_fold, k, logInfoFun=print):
        super().__init__(inputsDir, labelsPath, inputSuffix, K_fold, k, logInfoFun)
        self.m_massCenterDict = {}
        self.loadMassCenterForEachLabeledSlice()

    def loadMassCenterForEachLabeledSlice(self):
        massCenterFileName = "massCenterForEachLabeledSlice.json"
        filePath = os.path.join(self.m_inputsDir.replace("/images_npy", "/labels_npy"), massCenterFileName)
        if os.path.isfile(filePath):
            with open(filePath) as f:
                self.m_massCenterDict = json.load(f)
        else:
           self.m_logInfo(f"Error: program can not load {filePath}")
           sys.exit(-5)

    def dataResponseGenerator(self, inputFileIndices, shuffle=True, dataAugment=True, reSample=True):
        """
        yield (3DImage  - treatment Response) Tuple

        """
        shuffledList = inputFileIndices.copy()
        if reSample:
            shuffledList = self.reSampleForSameDistribution(shuffledList)
        if shuffle:
            random.shuffle(shuffledList)

        batch = 0
        dataList=[]  # for yield
        responseList= []

        # for crop ROIs
        imageGoalSize = (self.m_depth, self.m_height, self.m_width)  # (29, 140, 140)
        imageRadius = imageGoalSize[0] // 2

        for i in shuffledList:
            imageFile = self.m_inputFilesList[i]
            imageFileStem = self.getStemName(imageFile, self.m_inputSuffix)
            massCenterList = self.m_massCenterDict[imageFileStem]
            if dataAugment:
                massCenter = random.choice(massCenterList)
            else:
                massCenter = massCenterList[len(massCenterList) // 2]  # non dataAugment, choose the center labeled slice

            image3d = np.load(imageFile)

            # randomize ROI to generate the center of ROI
            z, x, y = massCenter
            if dataAugment:
                z = random.randrange(z - 6, z + 7, 1)  # the depth of image ROI is 145mm, max offset 20% = 29mm
                x = random.randrange(x - 28, x + 29, 1)  # the height of image ROI is 280mm, max offset 20% = 56mm
                y = random.randrange(y - 28, y + 29, 1)  # the width of image ROI is  280mm, max offset 20% = 56mm

            roiImage3d = self.cropVolumeCopyWithDstSize(image3d, z, x, y, imageRadius, imageGoalSize[1],  imageGoalSize[2])

            response = self.m_responseList[i]

            roiImage3d = self.preprocessData(roiImage3d)  # window level, and normalization.
            # data augmentation
            if dataAugment:
                roiImage3d = self.addGaussianNoise(roiImage3d)
                roiImage3d, roiSeg3d = self.flipDataLabel(roiImage3d, roiSeg3d)
                roiImage3d, roiSeg3d = self.rotate90s(roiImage3d,
                                                      roiSeg3d)  # around axis 0 rotation,and x=y so iamge size unchange.

            roiImage3d = np.expand_dims(roiImage3d, 0)  # add channel dim as 1
            dataList.append(roiImage3d)
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
        if 0 != len(dataList) and 0 != len(responseList): # PyTorch supports dynamic batchSize.
            yield np.stack(dataList, axis=0), np.stack(responseList, axis=0)

        # clean field
        dataList.clear()
        responseList.clear()

    def dataSegResponseGenerator(self, inputFileIndices, shuffle=True, convertAllZeroSlices=True, dataAugment=True, reSample=True):
        """
        yied (3DImage  -- Segmentation --  treatment Response) Tuple

        """
        shuffledList = inputFileIndices.copy()
        if reSample:
            shuffledList = self.reSampleForSameDistribution(shuffledList)
        if shuffle:
            random.shuffle(shuffledList)

        batch = 0
        dataList = []  # for yield
        segList = []
        responseList = []

        # for crop ROIs
        imageGoalSize = (self.m_depth, self.m_height, self.m_width)  #(29, 140, 140)
        labelGoalSize = (23, 127, 127)
        imageRadius = imageGoalSize[0] // 2
        labelRadius = labelGoalSize[0] // 2

        for i in shuffledList:
            imageFile = self.m_inputFilesList[i]
            imageFileStem = self.getStemName(imageFile, self.m_inputSuffix)
            massCenterList = self.m_massCenterDict[imageFileStem]
            if dataAugment:
                massCenter = random.choice(massCenterList)
            else:
                massCenter = massCenterList[len(massCenterList)//2]  # non dataAugment, choose the center labeled slice

            # for inputSize 147*281*281, and segmentation size of 127*255*255
            # labelFile = imageFile.replace("Images_ROI_29_140_140", "Labels_ROI_23_127_127")
            # labelFile = imageFile.replace("images_augmt_29_140_140", "labels_augmt_23_127_127")
            labelFile = imageFile.replace("/images_npy/", "/labels_npy/")  # the image and label are original various size

            image3d = np.load(imageFile)
            seg3d   = np.load(labelFile)

            # randomize ROI to generate the center of ROI
            z, x, y = massCenter
            if dataAugment:
                z = random.randrange(z-6, z+7, 1)   # the depth of image ROI is 145mm, max offset 20% = 29mm
                x = random.randrange(x-28, x+29, 1)   # the height of image ROI is 280mm, max offset 20% = 56mm
                y = random.randrange(y-28, y+29, 1)   # the width of image ROI is  280mm, max offset 20% = 56mm

            roiImage3d = self.cropVolumeCopyWithDstSize(image3d, z, x, y,  imageRadius, imageGoalSize[1], imageGoalSize[2])
            roiSeg3d = self.cropVolumeCopyWithDstSize(seg3d, z, x, y,  labelRadius, labelGoalSize[1], labelGoalSize[2])
            roi3 = roiSeg3d >= 3
            roiSeg3d[np.nonzero(roi3)] = 0  # erase label 3(lymph node)

            if convertAllZeroSlices:
                self.convertAllZeroSliceToValue(roiSeg3d, -100)  # -100 is default ignore_index in CrossEntropyLoss
            response = self.m_responseList[i]

            roiImage3d = self.preprocessData(roiImage3d)  # window level, and normalization.
            # data augmentation
            if dataAugment:
                roiImage3d = self.addGaussianNoise(roiImage3d)
                roiImage3d, roiSeg3d = self.flipDataLabel(roiImage3d, roiSeg3d)
                roiImage3d, roiSeg3d = self.rotate90s(roiImage3d, roiSeg3d)  # around axis 0 rotation,and x=y so iamge size unchange.

            roiImage3d = np.expand_dims(roiImage3d, 0)  # add channel dim as 1
            dataList.append(roiImage3d)
            segList.append(roiSeg3d)
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
        if 0 != len(dataList) and 0 != len(responseList): # PyTorch supports dynamic batchSize.
            yield np.stack(dataList, axis=0), np.stack(segList, axis=0), np.stack(responseList, axis=0)

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


