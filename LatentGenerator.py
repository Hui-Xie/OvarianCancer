
from DataMgr import DataMgr
from scipy import ndimage
import os
import numpy as np

class LatentGenerator(DataMgr):
    def __init__(self, inputsDir, labelsDir, inputSuffix, logInfoFun=print):
        super().__init__(inputsDir, labelsDir, inputSuffix, logInfoFun)
        self.createLatentDir()
        self.m_inputFilesList = self.getFilesList(self.m_inputsDir, self.m_inputSuffix)

    def sectionGenerator(self, imageFile, heightVolume):
        """

        :param imageFile:
        :param heightVolume: it is better as odd.
        :return:
        """
        labelFile = self.getLabelFile(imageFile)
        labelArray = self.readImageFile(labelFile)
        labelArray = (labelArray > 0)
        massCenterFloat = ndimage.measurements.center_of_mass(labelArray)
        massCenter = []
        for i in range(len(massCenterFloat)):
            massCenter.append(int(massCenterFloat[i]))

        imageArray = self.readImageFile(imageFile)
        numBatch = (heightVolume+self.m_batchSize-1)//self.m_batchSize

        d0 = massCenter[0] - heightVolume//2
        d0 = d0 if d0>=0 else 0
        hc = massCenter[1]
        wc = massCenter[2]
        for i in range(numBatch):
            d1 = d0 + i*self.m_batchSize
            d2 = d0 + (i+1)*self.m_batchSize
            if i == numBatch -1:
                d2 = d0+ heightVolume

            volume = self.cropContinuousVolume(imageArray, d1, d2, hc, wc)
            volume = np.expand_dims(volume, axis=1)
            yield volume
            if self.m_oneSampleTraining:
                break

    def createLatentDir(self):
        self.m_latentDir =  os.path.join(os.path.dirname(self.m_inputsDir), 'latent')
        if not os.path.exists(self.m_latentDir):
            os.mkdir(self.m_latentDir)

    def saveLatentV(self, array, patientID):
        np.save(os.path.join(self.m_latentDir, patientID+"_Latent.npy"), array)
