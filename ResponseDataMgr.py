from DataMgr import DataMgr
import json
import random
import os

class ResponseDataMgr(DataMgr):
    def __init__(self, inputsDir, labelsPath, inputSuffix, K_fold, k, logInfoFun=print):
        super().__init__(inputsDir, labelsPath, inputSuffix, K_fold, k, logInfoFun)
        self.m_responseList = []
        self.m_res0FileIndices = []
        self.m_res1FileIndices = []
        self.initializeInputsResponseList()


    def initializeInputsResponseList(self):
        if os.path.isfile(self.m_inputFilesListFile):
            self.loadInputFilesList()
        else:
            self.m_inputFilesList = self.getFilesList(self.m_inputsDir, self.m_inputSuffix)
            self.m_logInfo(f"program re-initializes all input files list, which will lead previous all K_fold cross validation invalid.")
            self.saveInputFilesList()

        self.m_logInfo(f"Now program get {len(self.m_inputFilesList)} input files.")
        self.m_responseList = []
        self.updateResponseList()
        self.statisticsReponse()
        self.divideTrainingValidationSet()

    def updateResponseList(self):
        with open(self.m_labelsDir) as f:
            allPatientRespsDict = json.load(f)

        for file in self.m_inputFilesList:
            patientID = self.getStemName(file, self.m_inputSuffix)
            if len(patientID) > 8:
                patientID = patientID[0:8]
            self.m_responseList.append(allPatientRespsDict[patientID])

    def getResponseCEWeight(self):
        labelPortion = [0.3, 0.7]  # this is portion of 0,1 label, whose sum = 1
        ceWeight = [0.0, 0.0]
        for i in range(2):
            ceWeight[i] = 1.0/labelPortion[i]

        self.m_logInfo(f"Infor: Response Cross Entropy Weight: {ceWeight} for label[0, 1]")
        return ceWeight

    def statisticsReponse(self):
        for i, res in enumerate(self.m_responseList):
            if res ==0:
                self.m_res0FileIndices.append(i)
            else: # res == 1
                self.m_res1FileIndices.append(i)
        self.m_logInfo(f"Infor: In all data of {len(self.m_responseList)} files, respone 0 has {len(self.m_res0FileIndices)} files,\n\t  and response 1 has {len(self.m_res1FileIndices)} files, "\
                       + f"where positive response rate = {len(self.m_res1FileIndices)/len(self.m_responseList)} in full data")

    def divideTrainingValidationSet(self):
        validationRate = 1.0/self.m_K_fold
        N0 = len(self.m_res0FileIndices)
        N1 = len(self.m_res1FileIndices)
        nV0 = int( N0* validationRate)
        nV1 = int( N1* validationRate)
        random.seed(201906)
        random.shuffle(self.m_res0FileIndices)
        random.shuffle(self.m_res1FileIndices)

        self.m_validationSetIndices = self.m_res0FileIndices[self.m_k*nV0:(self.m_k+1)*nV0]
        self.m_validationSetIndices += self.m_res1FileIndices[self.m_k*nV1:(self.m_k+1)*nV1]
        if self.m_k == 0:
            self.m_trainingSetIndices  = self.m_res0FileIndices[nV0:]
            self.m_trainingSetIndices  += self.m_res1FileIndices[nV1:]
        else:
            self.m_trainingSetIndices = self.m_res0FileIndices[0:self.m_k*nV0] + self.m_res0FileIndices[(self.m_k+1)*nV0:]
            self.m_trainingSetIndices += self.m_res1FileIndices[0:self.m_k*nV1] + self.m_res1FileIndices[(self.m_k+1)*nV1:]

        self.m_logInfo(f"Infor: Validation Set has {len(self.m_validationSetIndices)} files,and Training Set has {len(self.m_trainingSetIndices)} files")
        self.m_logInfo(f"Infor: Validataion set has {nV1} 1's, and positive response rate = {nV1/(nV0+ nV1)}")
        self.m_logInfo(f"Infor: trainning set has {N1-nV1} 1's, and positive response rate = {(N1-nV1)/(N0-nV0+ N1-nV1)}")
        self.m_logInfo(f"Infor: the drop_last data in the dataMgr may lead the number of validation set and training set less than above number.")

    def reSampleForSameDistribution(self, dataSetIndices):
        res0List = []
        res1List = []
        for x in dataSetIndices:
            res = self.m_responseList[x]
            if 0 == res:
                res0List.append(x)
            else:
                res1List.append(x)

        if len(res0List) < len(res1List):
            smallList = res0List
            bigList = res1List
        elif  len(res0List) > len(res1List):
            smallList = res1List
            bigList = res0List
        else:
            return dataSetIndices

        N = len(bigList) - len(smallList)
        pN = len(smallList)  # population N
        if N <= pN:
            reSamples = random.sample(smallList, N)
        else:
            reSamples = smallList*(N//pN) + random.sample(smallList, N%pN)

        return dataSetIndices+reSamples

