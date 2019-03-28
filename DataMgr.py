import os

class DataMgr:
    def __init__(self, imagesDir, labelsDir):
        self.m_imagesDir = imagesDir
        self.m_labelsDir = labelsDir

    def getFilesList(self, filesDir, suffix):
        return [os.path.abspath(x) for x in os.listdir(filesDir) if suffix in x]

