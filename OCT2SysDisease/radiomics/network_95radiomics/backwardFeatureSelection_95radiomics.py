# Sequential Backward feature selection on 95 radiomics

dataDir =  "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15_95radiomics"
ODOS = "ODOS"

# for sequential backward feature choose, use all data at CV0
trainingDataPath = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV/trainID_95radiomics_10CV_0.csv"
validationDataPath = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV/validationID_95radiomics_10CV_0.csv"
testDataPath = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV/testID_95radiomics_10CV_0.csv"

clinicalGTPath = "/home/hxie1/data/BES_3K/GTs/BESClinicalGT_Analysis.csv"

outputDir = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV"

imageSize =  [1,95]
numRadiomics = 95
keyName = "hypertension_bp_plus_history$"
srcSuffix = "_Volume_s15_95radiomics.npy"
class01Percent =  [0.4421085660495764,0.5578914339504236]  # according to HBP tag and xml image.

output2File = True

import glob
import sys
import os
import fnmatch
import numpy as np
from framework.measure import search_Threshold_Acc_TPR_TNR_Sum_WithProb
sys.path.append("..")
from OCT2SysD_Tools import readBESClinicalCsv

import datetime
import statsmodels.api as sm


output2File = True

def retrieveImageData_label(mode):
    '''
    :param mode: "training", "validation", or "test"

    '''
    if mode == "training":
        IDPath = trainingDataPath
    elif mode == "validation":
        IDPath = validationDataPath
    elif mode == "test":
        IDPath = testDataPath
    else:
        print(f"OCT2SysDiseaseDataSet mode error")
        assert False

    with open(IDPath, 'r') as idFile:
        IDList = idFile.readlines()
    IDList = [item[0:-1] for item in IDList]  # erase '\n'

    # get all correct volume numpy path
    allVolumesList = glob.glob(dataDir + f"/*{srcSuffix}")
    nonexistIDList = []
    multipleImages_IDList = []

    # make sure volume ID and volume path has strict corresponding order
    volumePaths = []  # number of volumes is about 2 times of IDList
    IDsCorrespondVolumes = []

    volumePathsFile = os.path.join(hps.dataDir, mode + f"_{hps.ODOS}_VolumePaths.txt")
    IDsCorrespondVolumesPathFile = os.path.join(hps.dataDir, mode + f"_{hps.ODOS}_IDsCorrespondVolumes.txt")

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
            resultList = fnmatch.filter(allVolumesList, "*/" + ID + f"_{hps.ODOS}_*{hps.volumeSuffix}")  # only for OD or OS data
            resultList.sort()
            numVolumes = len(resultList)
            if 0 == numVolumes:
                nonexistIDList.append(ID)
            elif numVolumes > 1:
                multipleImages_IDList.append(ID)
            else:
                volumePaths += resultList
                IDsCorrespondVolumes += [ID, ] # one ID, one volume

        if len(nonexistIDList) > 0:
            print(f"nonExistIDList of {hps.ODOS} in {mode}:\n {nonexistIDList}")
        if len(multipleImages_IDList) > 0:
            print(f"List of ID corresponding multiple {hps.ODOS} images in {mode}:\n {multipleImages_IDList}")

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
    # prepare output file
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(outputDir, f"95radiomics_FeatureSelection_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    print(f"===============Logistic Regression Feature Selection from 95 radiomics ================")

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

    # x = np.concatenate((volumes, clinicalFtrs), axis=1)
    # x = clinicalFtrs  # only use clinical features for sequential backward feature selection
    x = volumes   # only use thickness features for sequential backward feature selection
    y = labels[:, 1]  # hypertension
    x = np.delete(x, emptyRows, 0)
    y = np.delete(y, emptyRows, 0)
    print(f"After deleting empty-value patients, it remains {len(y)} patients.")

    # store the full feature names and its indexes in the x:
    fullFtrNames = []
    fullFtrIndexes = []
    index = 0
    for layer in range(hps.inputChannels):
        for sector in range(hps.imageH):
            fullFtrNames.append(f"L{layer}_S{sector}")
            fullFtrIndexes.append(index)
            index += 1
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
    print(f"selected feature indexes: {curIndexes}\n")

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

    print(f"========== End of 95-radiomics Features Selection ============ ")


if __name__ == "__main__":
    main()
