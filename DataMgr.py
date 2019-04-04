import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


class DataMgr:
    def __init__(self, imagesDir, labelsDir):
        self.m_imagesDir = imagesDir
        self.m_labelsDir = labelsDir
        self.m_oneSampleTraining = False

    def getFilesList(self, filesDir, suffix):
        originalCwd = os.getcwd()
        os.chdir(filesDir)
        filesList = [os.path.abspath(x) for x in os.listdir(filesDir) if suffix in x]
        os.chdir(originalCwd)
        return filesList

    def getTestDirs(self):
        return (self.m_imagesDir.replace('/trainImages', '/testImages'), self.m_labelsDir.replace('/trainLabels', '/testLabels'))

    def readImageFile(self, filename):
        image = sitk.ReadImage(filename)
        dataArray = sitk.GetArrayFromImage(image) #numpy axis order is a reverse of ITK axis order
        return dataArray

    def readImageAttributes(self, filename):
        image = sitk.ReadImage(filename)
        # these attributes are in ITK axis order
        self.m_origin = image.GetOrigin()
        self.m_size = image.GetSize()
        self.m_spacing = image.GetSpacing()
        self.m_direction = image.GetDirection()

    def saveImage(self, numpyArray, indexOffset, filename):
        '''
         saveImage from numpyArray
         SimpleITK and numpy indexing access is in opposite order!
        :param numpyArray:
        :param indexOffset: in numpy nd array axis order
        :param filename:
        :return:
        '''
        image = sitk.GetImageFromArray(numpyArray)
        offset = indexOffset.copy()[::-1]
        Dims = len(self.m_origin)
        origin = [self.m_origin[i]+ offset[i]*self.m_spacing[i]*self.m_direction[i*Dims+i] for i in range(Dims)]
        image.SetOrigin(origin)
        image.SetSpacing(self.m_spacing)
        image.SetDirection(self.m_direction)
        sitk.WriteImage(image, filename)
        print(f'File output: {filename} ')

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

    def cropVolumeCopy(self,array, dCenter, dRadius): # d means depth
        '''
        d means depth, we assume most axial images of patient are centered in its xy plane.
        :param array:
        :param dCenter:
        :param dRadius:
        :return:
        '''
        shape = array.shape

        d1 = dCenter-dRadius
        d1 = d1 if d1>=0 else 0
        d2 = d1+2*dRadius +1
        if d2 > shape[0]:
            d2 = shape[0]
            d1 = d2- 2*dRadius -1

        h1 = int((shape[1]-self.m_height)/2)
        h2 = h1+ self.m_height
        w1 = int((shape[2] - self.m_width) / 2)
        w2 = w1 + self.m_width

        return array[d1:d2, h1:h2, w1:w2].copy()

    def cropSliceCopy(self, array, dIndex):
        shape = array.shape
        h1 = int((shape[1] - self.m_height) / 2)
        h2 = h1 + self.m_height
        w1 = int((shape[2] - self.m_width) / 2)
        w2 = w1 + self.m_width
        return array[dIndex, h1:h2, w1:w2].copy()

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

    def getDiceSumList(self, outputs, labels):
        '''

        :param segmentations: with N samples
        :param labels: ground truth with N samples
        :return: a list, whose element 0 indicates total dice sum over N samples, element 1 indicate label1 dice sum over N samples, etc
        '''
        segmentations = self.oneHotArray2Segmentation(outputs)
        N = segmentations.shape[0]  # sample number
        K = self.m_k                # classification number
        diceSumList = [0 for _ in range(K)]
        for i in range(N):
            diceSumList[0] += self.getDice((segmentations[i] != 0) * 1, (labels[i] != 0) * 1)
            for j in range(1, K):
                diceSumList[j] += self.getDice((segmentations[i]==j)*1, (labels[i]==j)*1 )
        return diceSumList

    def getDice(self, segmentation, label):
        '''

        :param segmenatation:  0-1 elements array
        :param label:  0-1 elements array
        :return:
        '''
        nA = np.count_nonzero(segmentation)
        nB = np.count_nonzero(label)
        C = segmentation * label
        nC = np.count_nonzero(C)
        return nC*2.0/(nA+nB) if 0 != nA+nB else 1.0

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
        print(f'Input:  batchSize={self.m_batchSize}, depth={self.m_depth}, height={self.m_height}, width={self.m_width}, NumClassfication={self.m_k}')

    def getBatchSize(self):
        return self.m_batchSize

    def getInputSize(self): #return a tuple without batchSize
        channels = 1
        return (channels, self.m_depth, self.m_height, self.m_width)

    def getNumClassification(self):
        return self.m_k

    def dataLabelGenerator(self, shuffle):
        self.m_shuffle = shuffle
        imageFileList = self.getFilesList(self.m_imagesDir, "_CT.nrrd")
        N = len(imageFileList)
        shuffleList = list(range(N))
        if self.m_shuffle:
            random.shuffle(shuffleList)

        batch = 0
        dataList=[]
        labelList= []
        radius = int((self.m_depth-1)/2)

        for i in shuffleList:
            imageFile = imageFileList[i]
            #print(imageFile, f"i = {i}") # for debug
            labelFile = self.getLabelFile(imageFile)
            imageArray = self.readImageFile(imageFile)
            labelArray = self.readImageFile(labelFile)
            sliceList = self.getLabeledSliceIndex(labelArray)
            for j in sliceList:
                if batch >= self.m_batchSize:
                    yield np.stack(dataList, axis=0), np.stack(labelList, axis=0)
                    if self.m_oneSampleTraining:
                        continue
                    else:
                        batch = 0
                        dataList.clear()
                        labelList.clear()
                data = self.cropVolumeCopy(imageArray, j, radius)
                data = self.preprocessData(data)
                label= self.cropSliceCopy(labelArray,j)
                dataList.append(data)
                labelList.append(label)
                batch +=1
        # clean filed
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
        data = np.expand_dims(data, 0) # add channel dim as 1
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

    def setOneSampleTraining(self, oneSampleTrain):
        self.m_oneSampleTraining = oneSampleTrain





