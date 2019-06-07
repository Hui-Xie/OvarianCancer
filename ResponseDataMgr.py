from DataMgr import DataMgr
import json
import random

class ResponseDataMgr(DataMgr):
    def __init__(self, inputsDir, labelsPath, inputSuffix, logInfoFun=print):
        super().__init__(inputsDir, labelsPath, inputSuffix, logInfoFun)
        self.m_responseList = []
        self.initializeInputsResponseList()
        self.m_res0FileIndices = []
        self.m_res1FileIndices = []

    def initializeInputsResponseList(self):
        self.m_inputFilesList = self.getFilesList(self.m_inputsDir, self.m_inputSuffix)
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
        self.m_logInfo(f"Infor: Cross Entropy Weight: {ceWeight} for label[0, 1]")
        return ceWeight

    def statisticsReponse(self):
        for i, res in enumerate(self.m_responseList):
            if res ==0:
                self.m_res0FileIndices.append(i)
            else: # res == 1
                self.m_res1FileIndices.append(i)
        self.m_logInfo(f"Infor: In all data of {len(self.m_responseList)} files, respone 0 has {len(self.m_res0FileIndices)} and response 1 has {len(self.m_res1FileIndices)}")

    def divideTrainingValidationSet(self):
        validationRate = 0.2
        nValidation0 = int(len(self.m_res0FileIndices) * validationRate)
        nValidation1 = int(len(self.m_res1FileIndices) * validationRate)
        random.seed()
        random.shuffle(self.m_res0FileIndices)
        random.shuffle(self.m_res1FileIndices)
        self.m_validationSetIndices = self.m_res0FileIndices[0:nValidation0]
        self.m_validationSetIndices += self.m_res1FileIndices[0:nValidation1]
        self.m_trainingSetIndices  = self.m_res0FileIndices[nValidation0:]
        self.m_trainingSetIndices  += self.m_res1FileIndices[nValidation1:]
        self.m_logInfo(f"==== Regenerate training set and validation set by random with same distribution of 0 and 1 ==== ")
        self.m_logInfo(f"Infor: Validation Set has {len(self.m_validationSetIndices)} files,and Training Set has {len(self.m_trainingSetIndices)} files")

