import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


class DataMgr:
    def __init__(self, inputsDir, labelsDir, inputSuffix, logInfoFun=print):
        self.m_logInfo = logInfoFun
        self.m_oneSampleTraining = False
        self.m_inputsDir = inputsDir
        self.m_labelsDir = labelsDir
        self.m_inputSuffix = inputSuffix  # "_CT.nrrd", or "_zoom.npy", or "_roi.npy", or "_Latent.npy"

        self.m_alpha    = 0.4
        self.m_mixupProb = 0

        self.m_inputFilesList = []

        self.m_batchSize = 0
        self.m_depth = 0
        self.m_height = 0
        self.m_width = 0
        self.m_k = 0

        self.m_shuffle = True

    def setDataSize(self, batchSize, depth, height, width, k, dataName):
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
        self.m_k = k
        self.m_logInfo(
            f'{dataName} Input:  batchSize={self.m_batchSize}, depth={self.m_depth}, height={self.m_height}, width={self.m_width}, NumClassfication={self.m_k}\n')

    def getBatchSize(self):
        return self.m_batchSize

    def getInputSize(self):  # return a tuple without batchSize
        channels = 1
        if self.m_depth > 1:
            return channels, self.m_depth, self.m_height, self.m_width
        else:
            return channels, self.m_height, self.m_width

    def getNumClassification(self):
        return self.m_k

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


    @staticmethod
    def getFilesList(filesDir, suffix):
        originalCwd = os.getcwd()
        os.chdir(filesDir)
        filesList = [os.path.abspath(x) for x in os.listdir(filesDir) if suffix in x]
        os.chdir(originalCwd)
        return filesList

    def expandInputsDir(self, imagesDir, suffix):
        self.m_inputFilesList += self.getFilesList(imagesDir, suffix)
        self.m_logInfo(f'Expanding inputs dir: {imagesDir}')
        self.m_logInfo(f'Now dataMgr has {len(self.m_inputFilesList)} input files.')

    @staticmethod
    def readImageFile(filename):
        image = sitk.ReadImage(filename)
        dataArray = sitk.GetArrayFromImage(image)   # numpy axis order is a reverse of ITK axis order
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
            ptp = np.ptp(array, axesTuple) # peak to peak
            with np.nditer(ptp, op_flags=['readwrite']) as it:
                for x in it:
                    if 0 == x:
                        x[...] = 1e-6
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

    @staticmethod
    def getStemName(path, removedSuffix):
        baseName = os.path.basename(path)
        base = baseName[0: baseName.find(removedSuffix)]
        return base

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



