# analyze 9-sector thickness change over some risk factors.



import glob
import sys
import os
import fnmatch
import numpy as np
sys.path.append("../..")
from framework.ConfigReader import ConfigReader
sys.path.append("..")
from dataPrepare.OCT2SysD_Tools import readBESClinicalCsv

from scipy import stats
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import datetime
from scipy.stats import norm


def printUsage(argv):
    print("============ Anaylze OCT Thickness or texture map relation with hypertension =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_full_path")

def retrieveImageData_label(mode, hps):
    '''

    :param mode: "training", "validation", or "test"
    :param hps:
    :return: volumes: volumes of all patient in this data: NxHx1 for 9 sectors;
             labelTable:
    #labelTable head: patientID,                                          (0)
    #             "hypertension_bp_plus_history$", "gender", "Age$",'IOP$', 'AxialLength$', 'Height$', 'Weight$', 'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$'   (1:11)
    # columnIndex:         1                           2        3       4          5             6          7             8              9                10
    #              BMI, WHipRate,       (11,13)
    # columnIndex:  11    12

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
    volumes = np.empty((NVolumes, hps.imageH), dtype=float) # size:NxH for 9 sector array
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(float).squeeze(axis=-1)
        volumes[i,:] = oneVolume

    fullLabels = readBESClinicalCsv(hps.GTPath)

    labelTable = np.empty((NVolumes, 13), dtype=float) #  size: Nx11
    # table head: patientID,                                          (0)
    #             "hypertension_bp_plus_history$", "gender", "Age$",'IOP$', 'AxialLength$', 'Height$', 'Weight$', 'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$'   (1:11)
    # columnIndex:         1                           2        3       4          5             6          7             8              9                10
    #              BMI, WaistHipRate,       (11,13)
    # columnIndex:  11    12
    for i in range(NVolumes):
        id = int(IDsCorrespondVolumes[i])
        labelTable[i,0] = id

        # appKeys: ["hypertension_bp_plus_history$", "gender", "Age$",'IOP$', 'AxialLength$', 'Height$', 'Weight$', 'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$']
        for j,key in enumerate(hps.appKeys):
            oneLabel = fullLabels[id][key]
            if "gender" == key:
                oneLabel = oneLabel - 1
            labelTable[i, 1+j] = oneLabel

        # compute BMI and WHipRate
        labelTable[i, 11] = labelTable[i,7]/ ((labelTable[i,6]/100.0)**2)  # weight is in kg, height is in cm.
        labelTable[i, 12] = labelTable[i,8]/ labelTable[i,9]  # both are in cm.

    return volumes, labelTable


# refer to: https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
# may direct use statmodels: https://stackoverflow.com/questions/22306341/python-sklearn-how-to-calculate-p-values
def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    return: p values: where the 0th element is the p value of intercept.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se
    p = (1 - norm.cdf(abs(t))) * 2
    return p

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

        # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)

    # prepare output file
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
    nSectors = hps.imageH
    layerName= "5thThickness"


    # use thickness only to predict
    print("\n====== Use thickness only to predict hypertension =============")
    x = volumes
    y = labels[:, 1]  # hypertension
    print(f"Input only thickness, it has {len(y)} patients.")

    clf = LogisticRegression(max_iter=3000).fit(x, y)
    score = clf.score(x, y)
    pValues = logit_pvalue(clf, x)

    print(f"thickness only: score:{score}")
    print(f"Thickness coefficient of {layerName}:")
    print(f"intercept:{clf.intercept_[0]};\t p value: {pValues[0]}")
    for i in range(nSectors):
        print(f"thickness sector_{i}:\t{clf.coef_[0, i]};\t  p value: {pValues[i+1]}")

    print("\n====Use thickness and clinical feature to predict==========")
    appKeys = ["gender", "Age",'IOP', 'AxialLength','SmokePackYears', "BMI", "WaistHipRate",]
    appKeyColIndex = (2,3,4,5,10,11,12,)
    nClinicalFtr = len(appKeyColIndex)

    clinicalFtr = labels[:,appKeyColIndex]
    # delete the empty value of "-100"
    emptyRows = np.nonzero(clinicalFtr == -100)
    extraEmptyRows = np.nonzero(clinicalFtr == 99)
    emptyRows = (np.concatenate((emptyRows[0], extraEmptyRows[0]), axis=0),)
    # concatenate sector thickness with multi variables:
    thickness_features = np.concatenate((volumes, clinicalFtr), axis=1) # size: Nx(9+7)

    x = thickness_features
    y = labels[:, 1]  # hypertension
    x = np.delete(x, emptyRows, 0)
    y = np.delete(y, emptyRows, 0)
    print(f"After deleting empty-value patients, it remains {len(y)} patients.")

    clf = LogisticRegression(max_iter=3000).fit(x, y)
    score = clf.score(x, y)
    pValues = logit_pvalue(clf, x)

    print(f"thickness+7clinicalFeatures: score:{score}")
    print(f"Feature coefficient of {layerName}:")
    print(f"intercept:{clf.intercept_[0]};\t p value: {pValues[0]}")
    for i in range(nSectors+nClinicalFtr):
        if i<9:
            print(f"thickness sector_{i}:\t{clf.coef_[0,i]};\t p value: {pValues[i+1]} ")
        else:
            print(f"{appKeys[i-9]}:\t{clf.coef_[0,i]};\t p value: {pValues[i+1]}")

    logOutput.close()
    sys.stdout = original_stdout

    print(f"================ End of anlayzing 9-sector thickness  ===============")

if __name__ == "__main__":
    main()
