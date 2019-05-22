import os
import numpy as np
import random
import sys
from scipy.misc import imsave
from DataMgr import DataMgr

class SegDataMgr(DataMgr):
    def __init__(self, imagesDir, labelsDir, logInfoFun=print):
        super.__init__(imagesDir, labelsDir, logInfoFun)
        self.m_maxShift = 0
        self.m_translationProb = 0
        self.m_flipProb = 0
        self.m_rot90sProb = 0  # support 90, 180, 270 degree rotation
        self.m_noiseProb = 0  # noise probability
        self.m_noiseMean = 0
        self.m_noiseStd = 0
        self.m_jitterProb = 0
        self.m_jitterRadius = 0

        self.m_segDir = None
        self.m_segSliceTupleList = []
        self.m_imageAttrList = []

        self.m_binaryLabel = False
        self.m_remainedLabels = []  # the ground truth label value, for example 1,2,3
        self.m_suppressedLabels = []

        self.createSegmentedDir()
        self.m_imagesList = self.getFilesList(self.m_imagesDir, "_CT.nrrd")

    def setMaxShift(self, maxShift, translationProb = 0.5):
        self.m_maxShift = maxShift
        self.m_translationProb = translationProb

    def setFlipProb(self, prob):
        self.m_flipProb = prob

    def setRot90sProb(self, prob):
        self.m_rot90sProb = prob

    def setAddedNoise(self, prob, mean, std):
        self.m_noiseProb = prob
        self.m_noiseMean = mean
        self.m_noiseStd  = std

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

    def getCEWeight(self):
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
        self.m_logInfo(f"Infor: Cross Entropy Weight: {ceWeight}")
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
        for i, image in enumerate(self.m_imagesList):
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
        for image in self.m_imagesList:
            attr = self.getImageAttributes(image)
            self.m_imageAttrList.append(attr)

    def getTestDirs(self):  # may need to delete this function
        return self.m_imagesDir.replace('/trainImages', '/testImages'), self.m_labelsDir.replace('/trainLabels', '/testLabels')

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

    def getDiceSumList(self, segmentations, labels):
        """
        :param segmentations: with N samples
        :param labels: ground truth with N samples
        :return: (diceSumList,diceCountList)
                diceSumList: whose element 0 indicates total dice sum over N samples, element 1 indicate label1 dice sum over N samples, etc
                diceCountList: indicate effective dice count
        """
        N = segmentations.shape[0]  # sample number
        K = self.m_k                # classification number
        diceSumList = [0 for _ in range(K)]
        diceCountList = [0 for _ in range(K)]
        for i in range(N):
            (dice,count) = self.getDice((segmentations[i] != 0) * 1, (labels[i] != 0) * 1)
            diceSumList[0] += dice
            diceCountList[0] += count
            for j in range(1, K):
                (dice, count) = self.getDice((segmentations[i]==j)*1, (labels[i]==j)*1 )
                diceSumList[j] += dice
                diceCountList[j] += count

        return diceSumList, diceCountList

    @staticmethod
    def getDice(segmentation, label):
        """

        :param segmentation:  0-1 elements array
        :param label:  0-1 elements array
        :return: (dice, count) count=1 indicates it is an effective dice, count=0 indicates there is no nonzero elements in label.
        """
        nA = np.count_nonzero(segmentation)
        nB = np.count_nonzero(label)
        C = segmentation * label
        nC = np.count_nonzero(C)
        if 0 == nB:  # the dice was calculated over the slice where a ground truth was available.
            return 0, 0
        else:
            return nC*2.0/(nA+nB), 1

    @staticmethod
    def getTPR(segmentation, label):  #  sensitivity, recall, hit rate, or true positive rate (TPR)
        nB = np.count_nonzero(label)
        C = segmentation * label
        nC = np.count_nonzero(C)
        if 0 == nB:
            return 0, 0
        else:
            return nC/nB, 1

    def getTPRSumList(self, segmentations, labels):
        """
        :param segmentations: with N samples
        :param labels: ground truth with N samples
        :return: (TPRSumList,TPRCountList)
                TPRSumList: whose element 0 indicates total TPR sum over N samples, element 1 indicate label1 TPR sum over N samples, etc
                TPRCountList: indicate effective TPR count
        """
        N = segmentations.shape[0]  # sample number
        K = self.m_k                # classification number
        TPRSumList = [0 for _ in range(K)]
        TPRCountList = [0 for _ in range(K)]
        for i in range(N):
            (TPR,count) = self.getTPR((segmentations[i] != 0) * 1, (labels[i] != 0) * 1)
            TPRSumList[0] += TPR
            TPRCountList[0] += count
            for j in range(1, K):
                (TPR, count) = self.getTPR((segmentations[i]==j)*1, (labels[i]==j)*1 )
                TPRSumList[j] += TPR
                TPRCountList[j] += count

        return TPRSumList, TPRCountList

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

    def dataLabelGenerator(self, shuffle):
        """
        support 2D or 3D data shuffle
        :param shuffle: True or False
        :return:
        """
        self.m_shuffle = shuffle
        N = len(self.m_segSliceTupleList)
        shuffleList = list(range(N))
        if self.m_shuffle:
            random.shuffle(shuffleList)

        batch = 0
        dataList=[]  # for yield
        labelList= []
        radius = int((self.m_depth-1)/2)

        for n in shuffleList:
            (i,j) = self.m_segSliceTupleList[n]  # i is the imageID, j is the segmented slice index in image i.
            imageFile = self.m_imagesList[i]
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
                if self.m_oneSampleTraining:
                    continue
                else:
                    batch = 0
                    dataList.clear()
                    labelList.clear()

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

    def preprocessData(self, array)-> np.ndarray:
        data = array.clip(-300,300)    # adjust window level, also erase abnormal value
        data = self.sliceNormalize(data)
        data = self.jitterNoise(data)
        data = self.addGaussianNoise(data)
        return data

    def flipDataLabel(self, data, label):
        if self.m_flipProb >0 and random.uniform(0,1) <= self.m_flipProb:
            data  = np.flip(data, data.ndim-1)
            label = np.flip(label, label.ndim-1)
        return data, label

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

    def addGaussianNoise(self, data):
        if self.m_noiseProb >0 and random.uniform(0,1) <= self.m_noiseProb:
            noise = np.random.normal(self.m_noiseMean, self.m_noiseStd, data.shape)
            data += noise
        return data

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
            originalImagePath = self.m_imagesList[fileIndex]
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
    def oneHotArray2Segmentation(oneHotArray)-> np.ndarray:
        segmentationArray = oneHotArray.argmax(axis=1)
        return segmentationArray

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


    def updateDiceTPRSumList(self, outputsGPU, labelsCpu, diceSumList, diceCountList, TPRSumList, TPRCountList):
        outputs = outputsGPU.cpu().detach().numpy()
        segmentations = self.oneHotArray2Segmentation(outputs)

        (diceSumBatch, diceCountBatch) = self.getDiceSumList(segmentations, labelsCpu)
        (TPRSumBatch, TPRCountBatch) = self.getTPRSumList(segmentations, labelsCpu)

        diceSumList = [x + y for x, y in zip(diceSumList, diceSumBatch)]
        diceCountList = [x + y for x, y in zip(diceCountList, diceCountBatch)]
        TPRSumList = [x + y for x, y in zip(TPRSumList, TPRSumBatch)]
        TPRCountList = [x + y for x, y in zip(TPRCountList, TPRCountBatch)]

        return diceSumList, diceCountList, TPRSumList, TPRCountList