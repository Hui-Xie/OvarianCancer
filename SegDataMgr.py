import os
import numpy as np
import random
import sys
from scipy.misc import imsave
from DataMgr import DataMgr

class SegDataMgr(DataMgr):
    def __init__(self, inputsDir, labelsDir, inputSuffix, logInfoFun=print):
        super().__init__(inputsDir, labelsDir, inputSuffix, logInfoFun)
        self.m_maxShift = 0
        self.m_translationProb = 0



        self.m_jitterProb = 0
        self.m_jitterRadius = 0

        self.m_segDir = None
        self.m_segSliceTupleList = []
        self.m_imageAttrList = []

        self.m_binaryLabel = False
        self.m_remainedLabels = []  # the ground truth label value, for example 1,2,3
        self.m_suppressedLabels = []

        self.createSegmentedDir()
        self.m_inputFilesList = self.getFilesList(self.m_inputsDir, self.m_inputSuffix)

    def setMaxShift(self, maxShift, translationProb = 0.5):
        self.m_maxShift = maxShift
        self.m_translationProb = translationProb







    def setJitterNoise(self, prob, radius):
        self.m_jitterProb = prob
        self.m_jitterRadius = radius

    def setRemainedLabel(self, maxLabel, ks):
        if 0 in ks:
            self.m_remainedLabels = list(ks)
            self.m_suppressedLabels = self.getSuppressedLabels(maxLabel)
            if 2 == len(self.m_remainedLabels):
                self.m_binaryLabel = True
            self.m_logInfo(f"Infor: program test labels: {self.m_remainedLabels}")
            self.m_logInfo(f"Infor: program suppressed labels: {self.m_suppressedLabels}")

        else:
            self.m_logInfo(f"Error: background 0 should be in the remained label list")
            sys.exit(-1)

    def getSegCEWeight(self):
        labelPortion = [0.95995, 0.0254, 0.01462, 0.00003]  # this is portion of 0,1,2,3 label, whose sum = 1
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
        self.m_logInfo(f"Infor: Cross Entropy Weight: {ceWeight} for label {self.m_remainedLabels}")
        return ceWeight

    def createSegmentedDir(self):
        self.m_segDir = os.path.join(os.path.dirname(self.m_labelsDir), 'segmented')
        if not os.path.exists(self.m_segDir):
            os.mkdir(self.m_segDir)

    def getSuppressedLabels(self, maxLabel):
        labels = [x for x in range(maxLabel + 1)]
        result = labels[:]
        for x in labels:
            if x in self.m_remainedLabels:
                del result[result.index(x)]
        return result

    def buildSegSliceTupleList(self):
        """
        build segmented slice tuple list, in each tuple (fileID, segmentedSliceID)
        :return:
        """
        self.m_logInfo(f'Building the Segmented Slice Tuple list, which may need 8 mins, please waiting......')
        self.m_segSliceTupleList = []
        for i, image in enumerate(self.m_inputFilesList):
            label = self.getLabelFile(image)
            labelArray = self.readImageFile(label)
            sliceList = self.getLabeledSliceIndex(labelArray)
            for j in sliceList:
                self.m_segSliceTupleList.append((i, j))

            if self.m_oneSampleTraining and len(self.m_segSliceTupleList)>1:
                break
        self.m_logInfo(f'Directory of {self.m_labelsDir} has {len(self.m_segSliceTupleList)} segmented slices for remained labels {self.m_remainedLabels}.')

    def buildImageAttrList(self):
        """
        build a list of tuples including (origin, size, spacing, direction) in ITK axis order
        :return: void
        """
        self.m_logInfo(f"Building image attributes list, please waiting......")
        self.m_imageAttrList = []
        for image in self.m_inputFilesList:
            attr = self.getImageAttributes(image)
            self.m_imageAttrList.append(attr)



    def getLabeledSliceIndex(self,labelArray):
        labelArray = self.suppressedLabels(labelArray, binarize=False)
        nonzeroSlices = labelArray.sum((1, 2)).nonzero() # a tuple of arrays
        nonzeroSlices = nonzeroSlices[0]
        result = []
        if 0 == len(nonzeroSlices):
            return result
        previous = nonzeroSlices[0]
        start = previous
        for index in nonzeroSlices:
            if index- previous <= 1:
                previous = index
                continue
            else:
                result.append((previous+start)/2)
                start=previous=index
        result.append((previous + start) / 2)
        return [int(round(x, 0)) for x in result]





    def suppressedLabels(self, labelArray, binarize = True):
        for k in self.m_suppressedLabels:
            index = np.nonzero((labelArray == k)*1)
            labelArray[index] = 0

        if binarize and self.m_binaryLabel and not 1 in self.m_remainedLabels:
            index = np.nonzero(labelArray)
            labelArray[index] = 1

        return labelArray

    def randomTranslation(self,hc, wc):
        if self.m_maxShift > 0 and random.uniform(0,1) <= self.m_translationProb:
            hc += random.randrange(-self.m_maxShift, self.m_maxShift+1)
            wc += random.randrange(-self.m_maxShift, self.m_maxShift+1)
        return hc, wc

    def dataLabelGenerator(self, inputFileIndices, shuffle=True):
        """
        support 2D or 3D data shuffle
        :param shuffle: True or False
        :return:
        """
        shuffledList = inputFileIndices.copy()
        if shuffle:
            random.shuffle(shuffledList)

        batch = 0
        dataList=[]  # for yield
        labelList= []
        radius = int((self.m_depth-1)/2)

        for n in shuffledList:
            (i,j) = self.m_segSliceTupleList[n]  # i is the imageID, j is the segmented slice index in image i.
            imageFile = self.m_inputFilesList[i]
            labelFile = self.getLabelFile(imageFile)
            labelArray = self.readImageFile(labelFile)
            labelArrayJ = np.copy(labelArray[j])

            labelArrayJ = self.suppressedLabels(labelArrayJ, binarize=True)   # always erase label 3 as it only has 5 slices in dataset
            if 0 == np.count_nonzero(labelArrayJ):
                 continue

            imageArray = self.readImageFile(imageFile)
            imageArray, labelArrayJ = self.rotate90s(imageArray, labelArrayJ)  # rotation data augmentation

            (hc,wc) =  self.getLabelHWCenter(labelArrayJ) # hc: height center, wc: width center
            (hc,wc) = self.randomTranslation(hc, wc) # translation data augmentation

            if batch >= self.m_batchSize:
                yield np.stack(dataList, axis=0), np.stack(labelList, axis=0)
                batch = 0
                dataList.clear()
                labelList.clear()
                if self.m_oneSampleTraining:
                    break

            label = self.cropSliceCopy(labelArrayJ, hc, wc)
            if 0 == np.count_nonzero(label):   # skip the label without any meaningful labels
                continue

            if 0 != radius:
                data = self.cropVolumeCopy(imageArray, j, hc, wc, radius)
            else:
                data = self.cropSliceCopy(imageArray[j], hc, wc)

            data = self.preprocessData(data)

            (data, label) = self.flipDataLabel(data, label)

            data = np.expand_dims(data, 0)  # add channel dim as 1
            dataList.append(data)
            labelList.append(label)
            batch +=1

        if 0 != len(dataList) and 0 != len(labelList): # PyTorch supports dynamic batchSize.
            yield np.stack(dataList, axis=0), np.stack(labelList, axis=0)

        # clean field
        dataList.clear()
        labelList.clear()








    def jitterNoise(self, data):
        if self.m_jitterProb > 0 and self.m_jitterRadius >0  and  random.uniform(0, 1) <= self.m_jitterProb:
            ret = np.zeros(data.shape)
            dataIt = np.nditer(data, flags=['multi_index'])
            shape = data.shape
            while not dataIt.finished:
                index = dataIt.multi_index
                indexNew = self.indexDrift(index, shape, self.m_jitterRadius)
                ret[index] = data[indexNew]
                dataIt.iternext()
            return ret
        else:
            return data

    @staticmethod
    def indexDrift(index, shape, radius):
        ndim = len(shape)
        retIndex = ()
        for i in range(ndim):
            newIndex = index[i]+ random.randrange(-radius, radius+1, 1)
            if newIndex >= shape[i]:
                newIndex = shape[i]-1
            if newIndex <0:
                newIndex = 0
            retIndex +=(newIndex,)
        return retIndex

    def saveInputsSegmentations2Images(self, inputs, labels, segmentations, n):
        """

        :param inputs:
        :param labels:  ground truth
        :param segmentations:
        :param n:
        :return:
        """
        N = inputs.shape[0]
        for i in range(N):
            (fileIndex, sliceIndex) = self.m_segSliceTupleList[n+i]
            originalImagePath = self.m_inputFilesList[fileIndex]
            baseName = self.getStemName(originalImagePath, '_CT.nrrd')
            baseNamePath = os.path.join(self.m_segDir, baseName+f'_{sliceIndex}')
            inputImagePath = baseNamePath+ '.png'
            inputx = inputs[i]  # inputs shape:32,1,21,281,281
            if inputx.ndim ==4:
                inputSlice = inputx[0, int(inputx.shape[0]/2)]  # get the middle slice
            elif inputx.ndim ==3:
                inputSlice = inputx[0]
            else:
                self.m_logInfo(f"Error: image dimension eror in saveInputsSegmentations2Images")
                sys.exit(-1)
            imsave(inputImagePath, inputSlice)

            segImagePath = baseNamePath+ '_Seg.png'
            imsave(segImagePath, segmentations[i])

            groundtruthImagePath = baseNamePath + '_Label.png'
            imsave(groundtruthImagePath, labels[i])

            overlapSegImage = inputSlice +  segmentations[i]
            overlapSegPath = baseNamePath+ '_SegMerge.png'
            imsave(overlapSegPath,overlapSegImage)


            overlapLabelImage = inputSlice +  labels[i]
            overlapLabelPath = baseNamePath+ '_LabelMerge.png'
            imsave(overlapLabelPath,overlapLabelImage)



    @staticmethod
    def labelStatistic(labelArray, k):
        statisList = [0]*k
        for x in np.nditer(labelArray):
            statisList[x] +=1
        return statisList

    def batchLabelStatistic(self, batchLabelArray, k):
        N = batchLabelArray.shape[0]
        labelStatisSum = [0]*k
        sliceStatisSum = [0]*k
        for i in range(N):
            labelStatis = self.labelStatistic(batchLabelArray[i], k)
            for index, x in enumerate(labelStatis):
                if 0 !=x:
                    sliceStatisSum[index] +=1
            labelStatisSum = [x+y for x,y in zip(labelStatisSum, labelStatis)]
        return labelStatisSum, sliceStatisSum


