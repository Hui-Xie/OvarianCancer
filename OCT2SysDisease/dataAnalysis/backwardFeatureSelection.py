# Sequential Backward feature selection


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
import statsmodels.api as sm
# from sklearn.ensemble import RandomForestClassifier

output2File = True

def printUsage(argv):
    print("============ Sequential backward feature selection from 81 thickness and 14 clinical features =============")
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
    volumes = np.empty((NVolumes, hps.inputChannels, hps.imageH), dtype=np.float) # size:NxCxH for 9x9 sector array
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(np.float)
        volumes[i,:] = oneVolume
    volumes = volumes.reshape(-1,hps.inputChannels*hps.imageH* hps.imageW) # size: Nx(CxHxW)

    fullLabels = readBESClinicalCsv(hps.GTPath)

    labelTable = np.empty((NVolumes, 22), dtype=np.float) #  size: Nx22
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

    # concatinate all data crossing training, validation, and test
    volumes = np.concatenate((trainVolumes, validationVolumes, testVolumes), axis=0)
    labels = np.concatenate((trainLabels, validationLabels, testLabels), axis=0)

    print("\n== Sequential backward feature selection w.r.t. HBP ==============")
    assert len(hps.inputClinicalFeatures) == len(hps.clinicalFeatureColIndex)
    nClinicalFtr = len(hps.clinicalFeatureColIndex)
    clinicalFtrColIndex = tuple(hps.clinicalFeatureColIndex)
    inputClinicalFeatures = hps.inputClinicalFeatures


    clinicalFtrs = labels[:, clinicalFtrColIndex]

    # delete the empty value of "-100"
    emptyRows = np.nonzero((clinicalFtrs == -100).astype(np.int) + (clinicalFtrs < 0).astype(np.int))
    extraEmptyRows = np.nonzero(clinicalFtrs[:, inputClinicalFeatures.index("IOP")] == 99)  # IOP value
    emptyRows = (np.concatenate((emptyRows[0], extraEmptyRows[0]), axis=0),)

    x = np.concatenate((volumes, clinicalFtrs), axis=1)
    y = labels[:, 1]  # hypertension
    x = np.delete(x, emptyRows, 0)
    y = np.delete(y, emptyRows, 0)
    print(f"After deleting empty-value patients, it remains {len(y)} patients.")

    # store the full feature names and its indexes in the x:
    fullFtrNames=[]
    fullFtrIndexes = []
    index=0
    for layer in range(hps.inputChannels):
        for sector in range(hps.imageH):
            fullFtrNames.append(f"L{layer}_S{sector}")
            fullFtrIndexes.append(index)
            index +=1
    fullFtrNames += inputClinicalFeatures
    for i in range(index, index + len(inputClinicalFeatures)):
        fullFtrIndexes.append(i)
    assert len(fullFtrNames)==len(fullFtrIndexes)
    print(f"Initial input features before feature selection:\n{fullFtrNames}")
    print("")

    #================sequential backward feature selection========================
    print(f"============program is in sequential backward feature selection, please wait......==============")
    curIndexes = fullFtrIndexes.copy()
    curFtrs = fullFtrNames.copy()
    curClf = sm.Logit(y, x[:, tuple(curIndexes)]).fit(disp=0)
    curAIC = curClf.aic
    minAIC = curAIC
    predict = curClf.predict(x[:, tuple(curIndexes)])
    curAcc = np.mean((predict >= 0.5).astype(np.int) == y)
    print(f"number of features: {len(curIndexes)};\taic={minAIC};\tACC(cutoff0.5)={curAcc}")
    while True:
        # loop on each feature in current x to get aic for all delete feature
        isAICDecreased = False
        for i in range(0, len(curIndexes)):
            nextIndexes = curIndexes[0:i] + curIndexes[i+1:]
            nextClf = sm.Logit(y, x[:,tuple(nextIndexes)]).fit(disp=0)
            nextAIC = nextClf.aic
            if nextAIC < minAIC:
                minAIC = nextAIC
                minIndexes = nextIndexes
                minFtrs = curFtrs[0:i] + curFtrs[i + 1:]
                minClf  =nextClf
                isAICDecreased = True
        if isAICDecreased:
            curIndexes = minIndexes.copy()
            curFtrs = minFtrs.copy()
            predict = minClf.predict(x[:, tuple(curIndexes)])
            curAcc = np.mean((predict >= 0.5).astype(np.int) == y)
            print(f"number of features: {len(curIndexes)};\taic={minAIC};\tACC(cutoff0.5)={curAcc}")
        else:
            break

    print(f"========================End of sequential backward feature selection======================")
    print("Selected features with min AIC:")
    print(f"minAIC = {minAIC}")
    print(f"selected features: {curFtrs}")
    print(f"selccted feature indexes: {curIndexes}\n")

    #===Redo logistic regression with selected features===========
    clf = sm.Logit(y, x[:,tuple(curIndexes)]).fit(disp=0)
    print(clf.summary())
    predict = clf.predict(x[:,tuple(curIndexes)])
    accuracy = np.mean((predict >= 0.5).astype(np.int) == y)
    print(f"Accuracy of using {curFtrs} \n to predict hypertension with cutoff 0.5: {accuracy}")
    threhold_ACC_TPR_TNR_Sum = search_Threshold_Acc_TPR_TNR_Sum_WithProb(y, predict)
    print("With a different cut off with max(ACC+TPR+TNR):")
    print(threhold_ACC_TPR_TNR_Sum)
    print("Where:")
    for i in range(len(curFtrs)):
        print(f"x{i+1} = {curFtrs[i]}")

    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"================ End of sequential backward feature selection w.r.t. hypertension   ===============")

if __name__ == "__main__":
    main()
