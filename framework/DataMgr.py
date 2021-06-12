import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from utilities.FilesUtilities import *


class DataMgr:
    def __init__(self, inputsDir, labelsDir, inputSuffix, K_fold, k, logInfoFun=print):
        self.m_logInfo = logInfoFun
        self.m_oneSampleTraining = False
        self.m_inputsDir = inputsDir
        self.m_inputFilesListFile = os.path.join(self.m_inputsDir, "inputFilesList.txt")
        self.m_labelsDir = labelsDir
        self.m_inputSuffix = inputSuffix  # "_CT.nrrd", or "_zoom.npy", or "_roi.npy", or "_Latent.npy"

        self.m_K_fold = K_fold
        self.m_k = k

        self.m_alpha    = 0.4
        self.m_mixupProb = 0

        self.m_inputFilesList = []

        self.m_batchSize = 0
        self.m_depth = 0
        self.m_height = 0
        self.m_width = 0

        self.m_noiseProb = 0  # noise probability
        self.m_noiseMean = 0
        self.m_noiseStd = 0

        self.m_rot90sProb = 0  # support 90, 180, 270 degree rotation
        self.m_flipProb = 0

        self.m_trainingSetIndices = []
        self.m_validationSetIndices = []

    def setDataSize(self, batchSize, depth, height, width, dataName):
        """
        :param batchSize:
        :param depth:  it must be odd
        :param height:  it is better to be odd number for V model
        :param width:   it is better to be odd number for V model
        :param k: the number of classification of groundtruth including background class.
        :return:
        """
        self.m_batchSize = batchSize
        self.m_depth = depth
        self.m_height = height
        self.m_width = width
        self.m_logInfo(
            f'{dataName} Input:  batchSize={self.m_batchSize}, depth={self.m_depth}, height={self.m_height}, width={self.m_width}\n')

    def getBatchSize(self):
        return self.m_batchSize

    def getInputSize(self):  # return a tuple without batchSize
        channels = 1
        if self.m_depth > 1:
            return channels, self.m_depth, self.m_height, self.m_width
        else:
            return channels, self.m_height, self.m_width

    def setMixup(self, alpha, prob):
        self.m_alpha = alpha
        self.m_mixupProb = prob
        self.m_logInfo(f"Info: program uses Mixup with alpha={self.m_alpha}, and mixupProb = {self.m_mixupProb}.")

    def getLambdaInBeta(self):
        if self.m_mixupProb > 0 and random.uniform(0, 1) <= self.m_mixupProb:
            lambdaInBeta = np.random.beta(self.m_alpha, self.m_alpha)
        else:
            lambdaInBeta = 1.0
        return  lambdaInBeta

    def setAddedNoise(self, prob, mean, std):
        self.m_noiseProb = prob
        self.m_noiseMean = mean
        self.m_noiseStd = std

    def setRot90sProb(self, prob):
        self.m_rot90sProb = prob

    def setFlipProb(self, prob):
        self.m_flipProb = prob

    def expandInputsDir(self, imagesDir, suffix):
        self.m_inputFilesList += getFilesList(imagesDir, suffix)
        self.m_logInfo(f'Expanding inputs dir: {imagesDir}')
        self.m_logInfo(f'Now dataMgr has {len(self.m_inputFilesList)} input files.')

    @staticmethod
    def readImageFile(filename):
        image = sitk.ReadImage(filename)
        dataArray = sitk.GetArrayFromImage(image).astype(float)   # numpy axis order is a reverse of ITK axis order
        return dataArray

    @staticmethod
    def getImageAttributes(filename):
        """
        :param filename:
        :return: a tuple including (origin, size, spacing, direction) in ITK axis order
        """
        image = sitk.ReadImage(filename)
        # these attributes are in ITK axis order
        origin = image.GetOrigin()
        size = image.GetSize()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        return origin, size, spacing, direction

    @staticmethod
    def saveImage(imageAttr, numpyArray, indexOffset, filename):
        """
         saveImage from numpyArray
         SimpleITK and numpy indexing access is in opposite order!
        :param imageAttr, a tuple including (origin, size, spacing, direction) in ITK axis order
        :param numpyArray:
        :param indexOffset: in numpy nd array axis order
        :param filename:
        :return:
        """
        (origin, size, spacing, direction) = imageAttr
        image = sitk.GetImageFromArray(numpyArray)
        offset = indexOffset.copy()[::-1]
        Dims = len(origin)
        newOrigin = [origin[i] + offset[i]*spacing[i]*direction[i * Dims + i] for i in range(Dims)]
        image.SetOrigin(newOrigin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        sitk.WriteImage(image, filename)

    @staticmethod
    def getLabelFile(imageFile):
        return imageFile.replace("_CT.nrrd", "_Seg.nrrd").replace("Images/", "Labels/")


    @staticmethod
    def displaySlices(array, sliceList):
        N = len(sliceList)
        for i in range(N):
            plt.subplot(1,N, i+1)
            plt.imshow(array[sliceList[i],:,:])
        plt.show()

    def cropVolumeCopy(self,array, dc, hc, wc, dRadius): # d means depth
        """
        d means depth, we assume most axial images of patient are centered in its xy plane.
        :param array:
        :param dc: depth center
        :param hc: height center
        :param wc: width center
        :param dRadius:
        :return:
        """
        shape = array.shape

        d1 = dc-dRadius
        d1 = d1 if d1>=0 else 0
        d2 = d1+2*dRadius +1
        if d2 > shape[0]:
            d2 = shape[0]
            d1 = d2- 2*dRadius -1

        h1 = int(hc - self.m_height/2)
        h1 = h1 if h1>=0 else 0
        h2 = h1 + self.m_height
        if h2 > shape[1]:
            h2 = shape[1]
            h1 = h2- self.m_height

        w1 = int(wc - self.m_width / 2)
        w1 = w1 if w1 >= 0 else 0
        w2 = w1 + self.m_width
        if w2 > shape[2]:
            w2 = shape[2]
            w1 = w2 - self.m_width

        if  0 != dRadius:
            return array[d1:d2, h1:h2, w1:w2].copy()
        else:
            return array[d1:d2, h1:h2, w1:w2].copy().squeeze(axis=0)

    @staticmethod
    def cropVolumeCopyWithDstSize(array, dc, hc, wc, dRadius, dstHeight, dstWidth):  # d means depth
        """
        d means depth, we assume most axial images of patient are centered in its xy plane.
        :param array:
        :param dc: depth center
        :param hc: height center
        :param wc: width center
        :param dRadius:  = dstDepth Radius
        :return:
        """
        shape = array.shape

        d1 = dc - dRadius
        d1 = d1 if d1 >= 0 else 0
        d2 = d1 + 2 * dRadius + 1
        if d2 > shape[0]:
            d2 = shape[0]
            d1 = d2 - 2 * dRadius - 1

        h1 = int(hc - dstHeight / 2)
        h1 = h1 if h1 >= 0 else 0
        h2 = h1 + dstHeight
        if h2 > shape[1]:
            h2 = shape[1]
            h1 = h2 - dstHeight

        w1 = int(wc - dstWidth / 2)
        w1 = w1 if w1 >= 0 else 0
        w2 = w1 + dstWidth
        if w2 > shape[2]:
            w2 = shape[2]
            w1 = w2 - dstWidth

        if 0 != dRadius:
            return array[d1:d2, h1:h2, w1:w2].copy()
        else:
            return array[d1:d2, h1:h2, w1:w2].copy().squeeze(axis=0)


    def cropContinuousVolume(self, array, d1, d2, hc, wc):
        shape = array.shape

        h1 = int(hc - self.m_height / 2)
        h1 = h1 if h1 >= 0 else 0
        h2 = h1 + self.m_height
        if h2 > shape[1]:
            h2 = shape[1]
            h1 = h2 - self.m_height

        w1 = int(wc - self.m_width / 2)
        w1 = w1 if w1 >= 0 else 0
        w2 = w1 + self.m_width
        if w2 > shape[2]:
            w2 = shape[2]
            w1 = w2 - self.m_width

        return array[d1:d2, h1:h2, w1:w2].copy()

    def cropSliceCopy(self, slice, hc, wc):
        shape = slice.shape

        h1 = int(hc - self.m_height / 2)
        h1 = h1 if h1 >= 0 else 0
        h2 = h1 + self.m_height
        if h2 > shape[0]:
            h2 = shape[0]
            h1 = h2 - self.m_height

        w1 = int(wc - self.m_width / 2)
        w1 = w1 if w1 >= 0 else 0
        w2 = w1 + self.m_width
        if w2 > shape[1]:
            w2 = shape[1]
            w1 = w2 - self.m_width

        return slice[h1:h2, w1:w2].copy()

    @staticmethod
    def getLabelHWCenter(array2D):
        nonzerosIndex = np.nonzero(array2D)
        hc = int(nonzerosIndex[0].mean())
        wc = int(nonzerosIndex[1].mean())
        return hc,wc

    @staticmethod
    def segmentation2OneHotArray(segmentationArray, k)-> np.ndarray:
        """
        Convert segmentation volume to one Hot array used as ground truth in neural network
        :param segmentationArray:
        :param k:  number of classification including background 0
        :return:
        """
        shape = (k,)+segmentationArray.shape
        oneHotArray = np.zeros(shape)
        it = np.nditer(segmentationArray, flags=['multi_index'])
        while not it.finished:
            oneHotArray[(it[0],) + it.multi_index] = 1
            it.iternext()
        return oneHotArray

    def checkOrientConsistent(self, imagesDir, suffix):
        self.m_logInfo(f'Program is checking image directions. Please waiting......')
        imagesList = self.getFilesList(imagesDir, suffix)
        inconsistenNum = 0
        for filename in imagesList:
            image = sitk.ReadImage(filename)
            origin = image.GetOrigin()
            direction = image.GetDirection()
            Dims = len(origin)
            fullDirection = [direction[i]for i in range(Dims*Dims)]
            diagDirection = [direction[i * Dims + i]for i in range(Dims)]
            diagSum = sum(x>0 for x in diagDirection)
            if diagSum != 3:
                self.m_logInfo(f'{filename} has inconsistent direction: {fullDirection}')
                inconsistenNum +=1
        self.m_logInfo(f'Total {len(imagesList)} files, in which {inconsistenNum} files have inconsistent directions.')



    def sliceNormalize(self, array):
        if 3 == array.ndim:
            axesTuple = tuple([x for x in range(1, array.ndim)])
            minx = np.min(array, axesTuple)
            result = np.zeros(array.shape)
            for i in range(len(minx)):
                result[i,:] = array[i,:] - minx[i]
            ptp = list(np.ptp(array, axesTuple)) # peak to peak
            for i,x in enumerate(ptp):
                if x ==0:
                    ptp[i] = 1e-6
            for i in range(len(ptp)):
                result[i, :] /= ptp[i]
            return result
        elif 2 == array.ndim:
            minx = np.min(array)
            maxx = np.max(array)
            ptp = maxx-minx if maxx-minx != 0 else 1e-6
            result = (array - minx)/ptp
            return result
        else:
            self.m_logInfo("Error: the input to sliceNormalize has abnormal dimension.")
            sys.exit(0)


    def setOneSampleTraining(self, oneSampleTrain):
        self.m_oneSampleTraining = oneSampleTrain
        if self.m_oneSampleTraining:
            self.m_logInfo("Infor: program is in One Sample debug model.")
        else:
            self.m_logInfo("Infor: program is in multi samples running model.")


    @staticmethod
    def oneHotArray2Labels(oneHotArray) -> np.ndarray:
        labelsArray = oneHotArray.argmax(axis=1)
        return labelsArray

    def getTestDirs(self):  # may need to delete this function
        return self.m_inputsDir.replace('/trainImages', '/testImages'), self.m_labelsDir.replace('/trainLabels', '/testLabels')

    @staticmethod
    def convertAllZeroSliceToValue(segArray, toValue):
        nonzeroIndex = np.nonzero(segArray)
        nonzeroSlices  = set(nonzeroIndex[0])
        allSlices = set(range(segArray.shape[0]))
        zeroSlices = list(allSlices- nonzeroSlices)
        segArray[zeroSlices] = toValue


    @staticmethod
    def ignoreNegativeLabels(segmentations, labels):
        """
        convert the value in segmentations whose position correspond negtive label in labels to negtive.
        :param labels:
        :return:
        """
        negLabels = labels<0
        countNeg = np.count_nonzero(negLabels)
        if 0 == countNeg:
            return segmentations
        posiZeroLabels = labels>=0
        result = segmentations* ((negLabels*(-1)) + posiZeroLabels*1)
        return result

    def updateDiceTPRSumList(self, outputsGPU, labelsCpu, K, diceSumList, diceCountList, TPRSumList, TPRCountList):
        outputs = outputsGPU.detach().cpu().numpy()
        predictLabels= self.oneHotArray2Labels(outputs)

        predictLabels = DataMgr.ignoreNegativeLabels(predictLabels,labelsCpu)

        (diceSumBatch, diceCountBatch) = self.getDiceSumList(predictLabels, labelsCpu, K)
        (TPRSumBatch, TPRCountBatch) = self.getTPRSumList(predictLabels, labelsCpu, K)

        diceSumList = [x + y for x, y in zip(diceSumList, diceSumBatch)]
        diceCountList = [x + y for x, y in zip(diceCountList, diceCountBatch)]
        TPRSumList = [x + y for x, y in zip(TPRSumList, TPRSumBatch)]
        TPRCountList = [x + y for x, y in zip(TPRCountList, TPRCountBatch)]

        return diceSumList, diceCountList, TPRSumList, TPRCountList

    def preprocessData(self, array):
        data = array.clip(-300,300)    # adjust window level, also erase abnormal value
        data = self.sliceNormalize(data)
        return data

    def addGaussianNoise(self, data):
        if self.m_noiseProb >0 and random.uniform(0,1) <= self.m_noiseProb:
            noise = np.random.normal(self.m_noiseMean, self.m_noiseStd, data.shape)
            data += noise
        return data

    def rotate90s(self, data, label):
        if self.m_rot90sProb >0 and random.uniform(0,1) <= self.m_rot90sProb:
            k = random.randrange(1, 4, 1)  # k*90 is the real rotation degree
            data  = np.rot90(data, k, tuple(range(1,data.ndim)))
            if data.ndim == label.ndim:
                label = np.rot90(label, k, tuple(range(1,label.ndim)))
            elif data.ndim == label.ndim+1:
                label = np.rot90(label, k)
            else:
                self.m_logInfo("Error: in rotate90s, the ndim of data and label does not match.")
                sys.exi(-1)

        return data, label

    def flipDataLabel(self, data, label):
        if self.m_flipProb >0 and random.uniform(0,1) <= self.m_flipProb:
            data  = np.flip(data, data.ndim-1)
            label = np.flip(label, label.ndim-1)
        return data, label