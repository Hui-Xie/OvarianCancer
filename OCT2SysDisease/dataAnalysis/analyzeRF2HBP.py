# use random forest with 81+10 features to predict HBP(Hypertension)

import glob
import sys
import os
import fnmatch
import numpy as np
sys.path.append("../..")
from framework.ConfigReader import ConfigReader
from framework.measure import search_Threshold_Acc_TPR_TNR_Sum_WithProb
sys.path.append("..")
from dataPrepare.OCT2SysD_Tools import readBESClinicalCsv

from scipy import stats
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import datetime
# from scipy.stats import norm
# import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier

output2File = True

def printUsage(argv):
    print("============ Anaylze OCT 9x9 sector Thickness plus 10 risk factors to  predict hypertension =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_full_path")

def retrieveImageData_label(mode, hps):
    '''

    :param mode: "training", "validation", or "test"
    :param hps:
    :return:
             labelTable:
    #labelTable head: patientID,                                          (0)
    #             "hypertension_bp_plus_history$", "gender", "Age$",'IOP$', 'AxialLength$', 'Height$', 'Weight$', 'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$',   (1:11)
    # columnIndex:         1                           2        3       4          5             6          7             8              9                10
    #              'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',  'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015',
    # columnIndex:   11            12                           13                      14                       15                       16                  17
    #              'TG$_Corrected2015',  BMI,   WHipRate,  LDL/HDL
    #columnIndex:      18                 19       20         21

    '''
    if mode == "training":
        IDPath = hps.trainingDataPath
    elif mode == "validation":
        IDPath = hps.validationDataPath
    elif mode == "test":
        IDPath = hps.testDataPath
    else:
        print(f"OCT2SysDiseaseDataSet mode error")
        assert False

    with open(IDPath, 'r') as idFile:
        IDList = idFile.readlines()
    IDList = [item[0:-1] for item in IDList]  # erase '\n'

    # get all correct volume numpy path
    allVolumesList = glob.glob(hps.dataDir + f"/*{hps.volumeSuffix}")
    nonexistIDList = []

    # make sure volume ID and volume path has strict corresponding order
    volumePaths = []  # number of volumes is about 2 times of IDList
    IDsCorrespondVolumes = []

    volumePathsFile = os.path.join(hps.dataDir, mode + "_VolumePaths.txt")
    IDsCorrespondVolumesPathFile = os.path.join(hps.dataDir, mode + "_IDsCorrespondVolumes.txt")

    # save related file in order to speed up.
    if os.path.isfile(volumePathsFile) and os.path.isfile(IDsCorrespondVolumesPathFile):
        with open(volumePathsFile, 'r') as file:
            lines = file.readlines()
        volumePaths = [item[0:-1] for item in lines]  # erase '\n'

        with open(IDsCorrespondVolumesPathFile, 'r') as file:
            lines = file.readlines()
        IDsCorrespondVolumes = [item[0:-1] for item in lines]  # erase '\n'

    else:
        for i, ID in enumerate(IDList):
            resultList = fnmatch.filter(allVolumesList, "*/" + ID + f"_O[D,S]_*{hps.volumeSuffix}")
            resultList.sort()
            numVolumes = len(resultList)
            if 0 == numVolumes:
                nonexistIDList.append(ID)
            else:
                volumePaths += resultList
                IDsCorrespondVolumes += [ID, ] * numVolumes

        if len(nonexistIDList) > 0:
            print(f"Error: nonexistIDList:\n {nonexistIDList}")
            assert False

        # save files
        with open(volumePathsFile, "w") as file:
            for v in volumePaths:
                file.write(f"{v}\n")
        with open(IDsCorrespondVolumesPathFile, "w") as file:
            for v in IDsCorrespondVolumes:
                file.write(f"{v}\n")

    NVolumes = len(volumePaths)
    assert (NVolumes == len(IDsCorrespondVolumes))

    # load all volumes into memory
    assert hps.imageW == 1
    volumes = np.empty((NVolumes, hps.inputChannels, hps.imageH), dtype=float) # size:NxCxH for 9x9 sector array
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(float)
        volumes[i,:] = oneVolume
    volumes = volumes.reshape(-1,hps.inputChannels*hps.imageH* hps.imageW) # size: Nx(CxHxW)

    fullLabels = readBESClinicalCsv(hps.GTPath)

    labelTable = np.empty((NVolumes, 22), dtype=float) #  size: Nx22
    # labelTable head: patientID,                                          (0)
    #             "hypertension_bp_plus_history$", "gender", "Age$",'IOP$', 'AxialLength$', 'Height$', 'Weight$', 'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$',
    # columnIndex:         1                           2        3       4          5             6          7             8              9                10
    #              'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',  'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015',
    # columnIndex:   11            12                           13                      14                       15                       16                  17
    #              'TG$_Corrected2015',  BMI,   WaistHipRate,  LDL/HDL
    # columnIndex:      18                 19       20         21
    for i in range(NVolumes):
        id = int(IDsCorrespondVolumes[i])
        labelTable[i,0] = id

        #appKeys: ["hypertension_bp_plus_history$", "gender", "Age$", 'IOP$', 'AxialLength$', 'Height$', 'Weight$',
        #          'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$', 'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',
        #          'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015', 'TG$_Corrected2015']
        for j,key in enumerate(hps.appKeys):
            oneLabel = fullLabels[id][key]
            if "gender" == key:
                oneLabel = oneLabel - 1
            labelTable[i, 1+j] = oneLabel

        # compute BMI, WaistHipRate, LDL/HDL
        if labelTable[i,7]==-100 or labelTable[i,6] ==-100:
            labelTable[i, 19] = -100 # emtpty value
        else:
            labelTable[i, 19] = labelTable[i,7]/ ((labelTable[i,6]/100.0)**2)  # weight is in kg, height is in cm.

        if labelTable[i, 8] == -100 or labelTable[i, 9] == -100:
            labelTable[i, 20] = -100
        else:
            labelTable[i, 20] = labelTable[i,8]/ labelTable[i,9]  # both are in cm.

        if labelTable[i, 17] == -100 or labelTable[i, 16] == -100:
            labelTable[i, 21] = -100
        else:
            labelTable[i, 21] = labelTable[i, 17] / labelTable[i, 16]  # LDL/HDL, bigger means more risk to hypertension.

    return volumes, labelTable

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

        # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)

    # prepare output file
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(hps.outputDir, f"output_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    print(f"Experiment: {hps.experimentName}")

    # load training data, validation, and test data
    # volumes: volumes of all patient in this data: NxCxHxW;
    # labelTable: numpy array with columns: patientID, Hypertension(0/1), Age(value), gender(0/1);
    trainVolumes, trainLabels = retrieveImageData_label("training", hps)
    validationVolumes, validationLabels = retrieveImageData_label("validation", hps)
    testVolumes, testLabels = retrieveImageData_label("test", hps)

    print("\n== RandomForest using 81+10 features to predict HBP ==============")

    # concatenate 9x9 sector with clinical features, and delete empty-feature patients
    # 9 clinical features: ["Age", "IOP", "AxialLength", "Pulse", "Glucose", "Cholesterol", "Triglyceride", "BMI", "LDLoverHDL"]
    # 10 clinical features: ["gender", "Age", "IOP", "AxialLength", "Pulse", "Glucose", "Cholesterol", "Triglyceride", "BMI", "LDLoverHDL"]
    inputClinicalFeatures = hps.inputClinicalFeatures
    print(f"inputClinical features: {inputClinicalFeatures}")
    featureColIndex = tuple(hps.featureColIndex)
    nClinicalFtr = len(featureColIndex)
    assert nClinicalFtr == hps.numClinicalFtr

    volume_label_list= [
        ["train", trainVolumes, trainLabels,],
        ["validation", validationVolumes, validationLabels,],
        ["test", testVolumes, testLabels,],
    ]
    for name, volumes, labels in volume_label_list:
        clinicalFtrs = labels[:, featureColIndex]
        # delete the empty value of "-100"
        emptyRows = np.nonzero(clinicalFtrs == -100)
        extraEmptyRows = np.nonzero(clinicalFtrs[:, inputClinicalFeatures.index("IOP")] == 99)  # missing IOP value
        emptyRows = (np.concatenate((emptyRows[0], extraEmptyRows[0]), axis=0),)
        # concatenate sector thickness with multi variables:
        volumes = np.concatenate((volumes, clinicalFtrs), axis=1)  # size: Nx(81+len(self.m_inputClinicalFeatures))

        volumes = np.delete(volumes, emptyRows, 0)
        targetLabels = np.delete(labels, emptyRows, 0)[:, 1]  # for hypertension

        assert len(volumes) == len(targetLabels)
        print(f"size of {name} data set: {volumes.shape}")

        if name == "train":
            trainX = volumes
            trainY = targetLabels
        elif name == "validation":
            validationX = volumes
            validationY = targetLabels
        elif name =="test":
            testX = volumes
            testY = targetLabels
        else:
            print("Error data name")
            assert False

    # single Random Forest test
    clf = RandomForestClassifier(n_estimators=200, max_features=0.2)
    print("RandomForest: n_estimators=200,  max_features=0.2")
    print(f"Random forest parameters: \n{clf.get_params()}")
    clf.fit(trainX, trainY)
    trainAcc = clf.score(trainX, trainY)
    validationAcc = clf.score(validationX, validationY)
    testAcc = clf.score(testX, testY)

    print(f"training score: {trainAcc}")
    print(f"validation score: {validationAcc}")
    print(f"test score: {testAcc}")


    # Grid search best config for random forest
    print(f"\n================Grid search for Random Forest================")
    print(f"========the element in below table is validationACC_testAcc==========")
    print(f"=== the float feature indicate proportion of whole feature number====")
    RF_nFeatures = np.arange(0.1,0.32,0.02)
    RF_nEstimator = np.arange(100, 320, 20)
    strNFeatures = ", ".join([f"{elem:.2f}" for elem in RF_nFeatures])
    print(f"Estimators\Features, {strNFeatures}")
    for nEstimators in RF_nEstimator:
        print(nEstimators, end=", ")
        for nFeatures in RF_nFeatures:
            clf = RandomForestClassifier(n_estimators=nEstimators, max_features=nFeatures)
            clf.fit(trainX, trainY)
            validationAcc = clf.score(validationX, validationY)
            testAcc = clf.score(testX, testY)
            print(f"{validationAcc:.2f}_{testAcc:.2f}",end=", ")
        print("")
    print("==========================================================================")






    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"================ End of random forest from thickness and clinical features to hypertension   ===============")

if __name__ == "__main__":
    main()
