# analyze logisitic regression between fat factors and HBP.




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

output2File = True

def printUsage(argv):
    print("============ Anaylze OCT 9x9 sector Thickness plus risk factor with respect to hypertension =============")
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

    #== Logistic regression from Uni-variable to hypertension==============
    print("\n=================================================================")
    print("Logistic Regression between a clinical variable and hypertension:")
    variableKeys= ["gender", "Age", 'IOP', 'AxialLength', 'SmokePackYears', 'Pulse', 'Drink_quantity', 'Glucose', 'CRPL',
              'Cholesterol', 'HDL', 'LDL', 'Triglyceride', "BMI",   "WaistHipRate",  "LDL/HDL"]
    variableIndex = (2,3,4,5,10,11,12,13,14,15,16,17,18,19,20,21,)
    assert len(variableKeys) == len(variableIndex)

    for (keyIndex, colIndex) in enumerate(variableIndex):
        figureName = f"Hypertension_{variableKeys[keyIndex]}_logit"
        fig = plt.figure()

        y = labels[:, 1]  # hypertension
        x = labels[:, colIndex]

        # delete the empty value of "-100"
        emptyRows = np.nonzero((x < 0).astype(np.int) + (x==-100).astype(np.int))  # delete empty values, e.g -100
        if variableKeys[keyIndex] == "IOP":
            extraEmptyRows = np.nonzero(x == 99)
            emptyRows = (np.concatenate((emptyRows[0], extraEmptyRows[0]), axis=0),)
        x = np.delete(x, emptyRows, 0)
        y = np.delete(y, emptyRows, 0)

        plt.scatter(x, y, label='original data')

        # for single feature
        x = x.reshape(-1, 1)

        clf = sm.Logit(y, x).fit()
        print(f"\n===============Logistic regression between hypertension and {variableKeys[keyIndex]}===============")
        print(clf.summary())
        predict = clf.predict(x)
        accuracy = np.mean((predict >= 0.5).astype(np.int) == y)

        xtest = np.arange(x.min() * 0.95, x.max() * 1.05, (x.max() * 1.05 - x.min() * 0.95) / 100).reshape(-1, 1)
        plt.plot(xtest.ravel(), clf.predict(xtest).ravel(), 'r-', label='fitted line')
        textLocx = (x.max() * 1.05 - x.min() * 0.95)/2.2 + x.min()
        textLocy = 0.5
        plt.text(textLocx, textLocy, f"{accuracy:.1%}", fontsize=12)
        plt.xlabel(variableKeys[keyIndex])
        plt.ylabel(f"Hypertension")
        plt.legend()
        print(f"Accuracy of using {variableKeys[keyIndex]} to predict hypertension with cutoff 0.5: {accuracy}")

        outputFilePath = os.path.join(hps.outputDir, figureName + ".png")
        plt.savefig(outputFilePath)
        plt.close()

    print("\n\n====Multivariable Logistic regression between clinical risk factors and hypertension ==========")
    variableKeys = ["gender", "Age", 'IOP', 'AxialLength', 'SmokePackYears', 'Pulse', 'Drink_quantity', 'Glucose',
                    'CRPL', 'Cholesterol', 'Triglyceride', "BMI", "WaistHipRate", "LDL/HDL"]
    variableIndex = (2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21,)  # exclude HDL and LDL
    assert len(variableKeys) == len(variableIndex)
    nClinicalFtr = len(variableIndex)

    clinicalFtrs = labels[:,variableIndex]

    # delete the empty value of "-100"
    emptyRows = np.nonzero((clinicalFtrs == -100).astype(np.int) + (clinicalFtrs<0).astype(np.int))
    extraEmptyRows = np.nonzero(clinicalFtrs[:,2] == 99)  # IOP value
    emptyRows = (np.concatenate((emptyRows[0], extraEmptyRows[0]), axis=0),)

    x = clinicalFtrs
    y = labels[:, 1]  # hypertension
    x = np.delete(x, emptyRows, 0)
    y = np.delete(y, emptyRows, 0)
    print(f"After deleting empty-value patients, it remains {len(y)} patients.")

    clf = sm.Logit(y, x).fit()
    print(clf.summary())
    predict = clf.predict(x)
    accuracy = np.mean((predict >= 0.5).astype(np.int) == y)
    print(f"Accuracy of using {variableKeys} \n to predict hypertension with cutoff 0.5: {accuracy}")
    threhold_ACC_TPR_TNR_Sum = search_Threshold_Acc_TPR_TNR_Sum_WithProb(y, predict)
    print("With a different cut off:")
    print(threhold_ACC_TPR_TNR_Sum)
    print("Where:")
    n=1
    for i in range(nClinicalFtr):
        print(f"x{n}={variableKeys[i]}", end="; ")
        n += 1
    print("")
    print("=========================")
    print("list of x whose pvalue <=0.05")
    n = 1
    for i in range(0, nClinicalFtr):
        if clf.pvalues[n - 1] <= 0.05:
            print(f"x{n}={variableKeys[i]}, z={clf.tvalues[n - 1]}, pvalue={clf.pvalues[n - 1]}")
        n += 1


    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"================ End of logistic regression between clinical data and hypertension   ===============")

if __name__ == "__main__":
    main()
