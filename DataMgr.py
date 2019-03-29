import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


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
        return imageFile.replace("_CT.nrrd", "_Seg.nrrd").replace("/images/", "/labels/")

    def getLabeledSliceIndex(self, labelArray):
        nonzeroSlices = labelArray.sum((1,2)).nonzero(); # a tuple of arrays
        nonzeroSlices = nonzeroSlices[0]
        print(nonzeroSlices)
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

    def cropVolumeCopy(self,array, center, radius):
        return array[center-redius: center+radius+1,:,:].copy()

    def segmentation2OneHotArray(self, segmentationArray, k) -> np.ndarray:
        '''
        Convert segmenataion volume to one Hot array, used as ground truth in neural network
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



