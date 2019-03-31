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

    def getFilesList(self, filesDir, suffix):
        originalCwd = os.getcwd()
        os.chdir(filesDir)
        filesList = [os.path.abspath(x) for x in os.listdir(filesDir) if suffix in x]
        os.chdir(originalCwd)
        return filesList

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
        d1 = dCenter-dRadius
        d1 = d1 if d1>=0 else 0
        d2 = d1+2*dRadius +1
        shape = array.shape
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
        segmentationArray = oneHotArray.argmax(axis=0)
        return segmentationArray

    def setDataSize(self, batchSize, depth, height, width, k):
        '''
        :param batchSize:
        :param depth:  it must be odd
        :param height:  it is better to be odd number for V model
        :param width:   it is better to be odd number for V model
        :param k: the number of classification of groundtruth
        :return:
        '''
        self.m_batchSize = batchSize
        self.m_depth = depth
        self.m_height = height
        self.m_width = width
        self.m_k = k

    def dataLabelGenerator(self, shuffle):
        self.m_shuffle = shuffle
        imageFileList = self.getFilesList(self.m_imagesDir, "_CT.nrrd")
        N = len(imageFileList)
        shuffleList = list(range(N))
        if self.m_shuffle:
            random.shuffle(shuffleList)

        batch = 0
        dataList=[]
        oneHotLabelList= []
        radius = int((self.m_depth-1)/2)

        for i in shuffleList:
            imageFile = imageFileList[i]
            print(imageFile) # for debug
            labelFile = self.getLabelFile(imageFile)
            imageArray = self.readImageFile(imageFile)
            labelArray = self.readImageFile(labelFile)
            sliceList = self.getLabeledSliceIndex(labelArray)
            for j in sliceList:
                if batch >= self.m_batchSize:
                    yield np.stack(dataList, axis=0), np.stack(oneHotLabelList, axis=0)
                    batch = 0
                    dataList.clear()
                    oneHotLabelList.clear()
                data = self.cropVolumeCopy(imageArray, j, radius)
                label= self.cropSliceCopy(labelArray,j)
                oneHotLabel = self.segmentation2OneHotArray(label, self.m_k)
                dataList.append(data)
                oneHotLabelList.append(oneHotLabel)
                batch +=1

        # clean filed
        dataList.clear()
        oneHotLabelList.clear()

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









