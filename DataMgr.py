import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from scipy.misc import imsave


class DataMgr:
    def __init__(self, imagesDir, labelsDir):
        self.m_oneSampleTraining = False
        self.m_imagesDir = imagesDir
        self.m_labelsDir = labelsDir

        self.m_maxShift  = 0
        self.m_flipProb = 0

        self.buildSegSliceTupleList()
        self.createSegmentedDir()

    def setMaxShift(self, maxShift):
        self.m_maxShift = maxShift

    def setFlipProb(self, prob):
        self.m_flipProb = prob

    def createSegmentedDir(self):
        self.m_segDir =  os.path.join(os.path.dirname(self.m_labelsDir), 'segmented')
        if not os.path.exists(self.m_segDir):
            os.mkdir(self.m_segDir)

    def getFilesList(self, filesDir, suffix):
        originalCwd = os.getcwd()
        os.chdir(filesDir)
        filesList = [os.path.abspath(x) for x in os.listdir(filesDir) if suffix in x]
        os.chdir(originalCwd)
        return filesList

    def buildSegSliceTupleList(self):
        '''
        build segmented slice tuple list, in each tuple (fileID, segmentedSliceID)
        :return:
        '''
        print('Building the Segmented Slice Tuple list, please waiting......')
        self.m_segSliceTupleList = []
        self.m_imagesList = self.getFilesList(self.m_imagesDir, "_CT.nrrd")
        for i, image in enumerate(self.m_imagesList):
            label = self.getLabelFile(image)
            labelArray = self.readImageFile(label)
            sliceList = self.getLabeledSliceIndex(labelArray)
            for j in sliceList:
                self.m_segSliceTupleList.append((i,j))
        print(f'Directory of {self.m_labelsDir} has {len(self.m_segSliceTupleList)} segmented slices.')

    def buildImageAttrList(self):
        '''
        build a list of tuples including (origin, size, spacing, direction) in ITK axis order
        :return: void
        '''
        print("Building image attributes list, please waiting......")
        self.m_imageAttrList = []
        for image in self.m_imagesList:
            attr = self.getImageAttributes(image)
            self.m_imageAttrList.append(attr)

    def getTestDirs(self):  # may need to delete this function
        return (self.m_imagesDir.replace('/trainImages', '/testImages'), self.m_labelsDir.replace('/trainLabels', '/testLabels'))

    def readImageFile(self, filename):
        image = sitk.ReadImage(filename)
        dataArray = sitk.GetArrayFromImage(image) #numpy axis order is a reverse of ITK axis order
        return dataArray

    def getImageAttributes(self, filename):
        '''
        :param filename:
        :return: a tuple including (origin, size, spacing, direction) in ITK axis order
        '''
        image = sitk.ReadImage(filename)
        # these attributes are in ITK axis order
        origin = image.GetOrigin()
        size = image.GetSize()
        spacing = image.GetSpacing()
        direction = image.GetDirection()
        return (origin, size, spacing, direction)

    def saveImage(self, imageAttr, numpyArray, indexOffset, filename):
        '''
         saveImage from numpyArray
         SimpleITK and numpy indexing access is in opposite order!
        :param imageAttr, a tuple including (origin, size, spacing, direction) in ITK axis order
        :param numpyArray:
        :param indexOffset: in numpy nd array axis order
        :param filename:
        :return:
        '''
        (origin, size, spacing, direction) = imageAttr
        image = sitk.GetImageFromArray(numpyArray)
        offset = indexOffset.copy()[::-1]
        Dims = len(origin)
        newOrigin = [origin[i]+ offset[i]*spacing[i]*direction[i*Dims+i] for i in range(Dims)]
        image.SetOrigin(newOrigin)
        image.SetSpacing(spacing)
        image.SetDirection(direction)
        sitk.WriteImage(image, filename)


    def getLabelFile(self, imageFile):
        return imageFile.replace("_CT.nrrd", "_Seg.nrrd").replace("Images/", "Labels/")

    def getLabeledSliceIndex(self, labelArray):
        nonzeroSlices = labelArray.sum((1,2)).nonzero(); # a tuple of arrays
        nonzeroSlices = nonzeroSlices[0]
        if 0 == len(nonzeroSlices):
            print("Infor: label file does not find any nonzero label. ")
            sys.exit()
        result = []
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
        return [int(round(x,0)) for x in result]

    def displaySlices(self, array, sliceList):
        N = len(sliceList)
        for i in range(N):
            plt.subplot(1,N, i+1)
            plt.imshow(array[sliceList[i],:,:])
        plt.show()

    def cropVolumeCopy(self,array, dc, hc, wc, dRadius): # d means depth
        '''
        d means depth, we assume most axial images of patient are centered in its xy plane.
        :param array:
        :param dc: depth center
        :param hc: height center
        :param wc: width center
        :param dRadius:
        :return:
        '''
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

        return array[d1:d2, h1:h2, w1:w2].copy()

    def cropSliceCopy(self, array, dIndex, hc, wc):
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

        return array[dIndex, h1:h2, w1:w2].copy()

    def getLabelHWCenter(self, array2D):
        nonzerosIndex = np.nonzero(array2D)
        hc = int(nonzerosIndex[0].mean())
        wc = int(nonzerosIndex[1].mean())
        return (hc,wc)

    def segmentation2OneHotArray(self, segmentationArray, k) -> np.ndarray:
        '''
        Convert segmentation volume to one Hot array used as ground truth in neural network
        :param segmentationArray:
        :param k:  number of classification including background 0
        :return:
        '''
        shape = (k,)+segmentationArray.shape
        oneHotArray = np.zeros(shape)
        it = np.nditer(segmentationArray, flags=['multi_index'])
        while not it.finished:
             oneHotArray[(it[0],) + it.multi_index] = 1
             it.iternext()
        return oneHotArray

    def oneHotArray2Segmentation(self, oneHotArray) -> np.ndarray:
        segmentationArray = oneHotArray.argmax(axis=1)
        return segmentationArray

    def getDiceSumList(self, segmentations, labels):
        '''
        :param segmentations: with N samples
        :param labels: ground truth with N samples
        :return: (diceSumList,diceCountList)
                diceSumList: whose element 0 indicates total dice sum over N samples, element 1 indicate label1 dice sum over N samples, etc
                diceCountList: indicate effective dice count
        '''
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

        return (diceSumList, diceCountList)

    def getDice(self, segmentation, label):
        '''

        :param segmenatation:  0-1 elements array
        :param label:  0-1 elements array
        :return: (dice, count) count=1 indicates it is an effective dice, count=0 indicates there is no nonzero elements in label.
        '''
        nA = np.count_nonzero(segmentation)
        nB = np.count_nonzero(label)
        C = segmentation * label
        nC = np.count_nonzero(C)
        if 0 == nA+nB:
            return (0, 0)
        else:
            return (nC*2.0/(nA+nB), 1)

    def setDataSize(self, batchSize, depth, height, width, k):
        '''
        :param batchSize:
        :param depth:  it must be odd
        :param height:  it is better to be odd number for V model
        :param width:   it is better to be odd number for V model
        :param k: the number of classification of groundtruth including background class.
        :return:
        '''
        self.m_batchSize = batchSize
        self.m_depth = depth
        self.m_height = height
        self.m_width = width
        self.m_k = k
        print(f'Input:  batchSize={self.m_batchSize}, depth={self.m_depth}, height={self.m_height}, width={self.m_width}, NumClassfication={self.m_k}\n')

    def getBatchSize(self):
        return self.m_batchSize

    def getInputSize(self): #return a tuple without batchSize
        channels = 1
        return (channels, self.m_depth, self.m_height, self.m_width)

    def getNumClassification(self):
        return self.m_k

    def randomTranslation(self,hc, wc):
        if self.m_maxShift > 0:
            hc += random.randrange(-self.m_maxShift, self.m_maxShift+1)
            wc += random.randrange(-self.m_maxShift, self.m_maxShift+1)
        return (hc, wc)

    def dataLabelGenerator(self, shuffle):
        self.m_shuffle = shuffle
        N = len(self.m_segSliceTupleList)
        shuffleList = list(range(N))
        if self.m_shuffle:
            random.shuffle(shuffleList)

        batch = 0
        dataList=[]
        labelList= []
        radius = int((self.m_depth-1)/2)

        for n in shuffleList:
            (i,j) = self.m_segSliceTupleList[n]  # i is the imageID, j is the segmented slice index in image i.
            imageFile = self.m_imagesList[i]
            labelFile = self.getLabelFile(imageFile)
            imageArray = self.readImageFile(imageFile)
            labelArray = self.readImageFile(labelFile)
            (hc,wc) =  self.getLabelHWCenter(labelArray[j]) # hc: height center, wc: width center
            (hc,wc) = self.randomTranslation(hc, wc) # translation data augmentation

            if batch >= self.m_batchSize:
                yield np.stack(dataList, axis=0), np.stack(labelList, axis=0)
                if self.m_oneSampleTraining:
                    continue
                else:
                    batch = 0
                    dataList.clear()
                    labelList.clear()
            data = self.cropVolumeCopy(imageArray, j, hc, wc, radius)
            data = self.preprocessData(data)
            label= self.cropSliceCopy(labelArray,j, hc, wc)

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

    def checkOrientConsistent(self, imagesDir, suffix):
        print(f'Program is checking image directions. Please waiting......')
        imagesList = self.getFilesList(imagesDir, suffix)
        inconsistenNum = 0;
        for filename in imagesList:
            image = sitk.ReadImage(filename)
            origin = image.GetOrigin()
            direction = image.GetDirection()
            Dims = len(origin)
            fullDirection = [direction[i]for i in range(Dims*Dims)]
            diagDirection = [direction[i * Dims + i]for i in range(Dims)]
            diagSum = sum(x>0 for x in diagDirection)
            if diagSum != 3:
                print(f'{filename} has inconsistent direction: {fullDirection}')
                inconsistenNum +=1
        print(f'Total {len(imagesList)} files, in which {inconsistenNum} files have inconsistent directions.')

    def preprocessData(self, array) -> np.ndarray:
        data = array.clip(-300,300)
        data = self.sliceNormalize(data)
        return data

    def sliceNormalize(self, array):
        axesTuple = tuple([x for x in range(1, len(array.shape))])
        min = np.min(array, axesTuple)
        result = np.zeros(array.shape)
        for i in range(len(min)):
            result[i,:] = array[i,:] - min[i]
        ptp = np.ptp(array, axesTuple) # peak to peak
        with np.nditer(ptp, op_flags=['readwrite']) as it:
             for x in it:
                 x = x if 0!=x else 1e-6
        for i in range(len(ptp)):
            result[i, :] /= ptp[i]
        return result

    def flipDataLabel(self, data, label):
        if self.m_flipProb >0 and random.uniform(0,1) <= self.m_flipProb:
            data  = np.flip(data, len(data.shape)-1)
            label = np.flip(label, len(label.shape)-1)
        return (data, label)

    def setOneSampleTraining(self, oneSampleTrain):
        self.m_oneSampleTraining = oneSampleTrain

    def getStemName(self, path, removedSuffix):
        baseName = os.path.basename(path)
        base = baseName[0: baseName.find(removedSuffix)]
        return base

    def saveInputsSegmentations2Images(self, inputs, labels, segmentations, n):
        '''

        :param inputs:
        :param labels:  ground truth
        :param segmentations:
        :param n:
        :return:
        '''
        N = inputs.shape[0]
        for i in range(N):
            (fileIndex, sliceIndex) = self.m_segSliceTupleList[n+i]
            originalImagePath = self.m_imagesList[fileIndex]
            baseName = self.getStemName(originalImagePath, '_CT.nrrd')
            baseNamePath = os.path.join(self.m_segDir, baseName+f'_{sliceIndex}')
            inputImagePath = baseNamePath+ '.png'
            input = inputs[i]  # inputs shape:32,1,21,281,281
            inputSlice = input[0, int(input.shape[0]/2)]  # get the middle slice
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