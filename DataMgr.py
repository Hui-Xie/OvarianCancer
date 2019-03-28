import os
import SimpleITK as sitk
import matplotlib.pyplot as plt


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