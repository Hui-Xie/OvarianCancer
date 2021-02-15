# Sequential Backward feature selection on 95 radiomics

dataDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15_95radiomics"
ODOS = "ODOS"

radiomics95Features = [
    "original_firstorder_10Percentile",
    "original_firstorder_90Percentile",
    "original_firstorder_Energy",
    "original_firstorder_Entropy",
    "original_firstorder_InterquartileRange",
    "original_firstorder_Kurtosis",
    "original_firstorder_Maximum",
    "original_firstorder_Mean",
    "original_firstorder_MeanAbsoluteDeviation",
    "original_firstorder_Median",
    "original_firstorder_Minimum",
    "original_firstorder_Range",
    "original_firstorder_RobustMeanAbsoluteDeviation",
    "original_firstorder_RootMeanSquared",
    "original_firstorder_Skewness",
    "original_firstorder_TotalEnergy",
    "original_firstorder_Uniformity",
    "original_firstorder_Variance",
    "original_glcm_Autocorrelation",
    "original_glcm_ClusterProminence",
    "original_glcm_ClusterShade",
    "original_glcm_ClusterTendency",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_DifferenceAverage",
    "original_glcm_DifferenceEntropy",
    "original_glcm_DifferenceVariance",
    "original_glcm_Id",
    "original_glcm_Idm",
    "original_glcm_Idmn",
    "original_glcm_Idn",
    "original_glcm_Imc1",
    "original_glcm_Imc2",
    "original_glcm_InverseVariance",
    "original_glcm_JointAverage",
    "original_glcm_JointEnergy",
    "original_glcm_JointEntropy",
    "original_glcm_MaximumProbability",
    "original_glcm_SumEntropy",
    "original_glcm_SumSquares",
    "original_gldm_DependenceEntropy",
    "original_gldm_DependenceNonUniformity",
    "original_gldm_DependenceNonUniformityNormalized",
    "original_gldm_DependenceVariance",
    "original_gldm_GrayLevelNonUniformity",
    "original_gldm_GrayLevelVariance",
    "original_gldm_HighGrayLevelEmphasis",
    "original_gldm_LargeDependenceEmphasis",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis",
    "original_gldm_LargeDependenceLowGrayLevelEmphasis",
    "original_gldm_LowGrayLevelEmphasis",
    "original_gldm_SmallDependenceEmphasis",
    "original_gldm_SmallDependenceHighGrayLevelEmphasis",
    "original_gldm_SmallDependenceLowGrayLevelEmphasis",
    "original_glrlm_GrayLevelNonUniformity",
    "original_glrlm_GrayLevelNonUniformityNormalized",
    "original_glrlm_GrayLevelVariance",
    "original_glrlm_HighGrayLevelRunEmphasis",
    "original_glrlm_LongRunEmphasis",
    "original_glrlm_LongRunHighGrayLevelEmphasis",
    "original_glrlm_LongRunLowGrayLevelEmphasis",
    "original_glrlm_LowGrayLevelRunEmphasis",
    "original_glrlm_RunEntropy",
    "original_glrlm_RunLengthNonUniformity",
    "original_glrlm_RunLengthNonUniformityNormalized",
    "original_glrlm_RunPercentage",
    "original_glrlm_RunVariance",
    "original_glrlm_ShortRunEmphasis",
    "original_glrlm_ShortRunHighGrayLevelEmphasis",
    "original_glrlm_ShortRunLowGrayLevelEmphasis",
    "original_glszm_GrayLevelNonUniformity",
    "original_glszm_GrayLevelNonUniformityNormalized",
    "original_glszm_GrayLevelVariance",
    "original_glszm_HighGrayLevelZoneEmphasis",
    "original_glszm_LargeAreaEmphasis",
    "original_glszm_LargeAreaHighGrayLevelEmphasis",
    "original_glszm_LargeAreaLowGrayLevelEmphasis",
    "original_glszm_LowGrayLevelZoneEmphasis",
    "original_glszm_SizeZoneNonUniformity",
    "original_glszm_SizeZoneNonUniformityNormalized",
    "original_glszm_SmallAreaEmphasis",
    "original_glszm_SmallAreaHighGrayLevelEmphasis",
    "original_glszm_SmallAreaLowGrayLevelEmphasis",
    "original_glszm_ZoneEntropy",
    "original_glszm_ZonePercentage",
    "original_glszm_ZoneVariance",
    "original_shape2D_Elongation",
    "original_shape2D_MajorAxisLength",
    "original_shape2D_MaximumDiameter",
    "original_shape2D_MeshSurface",
    "original_shape2D_MinorAxisLength",
    "original_shape2D_Perimeter",
    "original_shape2D_PerimeterSurfaceRatio",
    "original_shape2D_PixelSurface",
    "original_shape2D_Sphericity",
]

# for sequential backward feature choose, use all data at CV0
trainingDataPath = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV/trainID_95radiomics_10CV_0.csv"
validationDataPath = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV/validationID_95radiomics_10CV_0.csv"
testDataPath = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV/testID_95radiomics_10CV_0.csv"

clinicalGTPath = "/home/hxie1/data/BES_3K/GTs/BESClinicalGT_Analysis.csv"

outputDir = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV"

# input 95 radiomics size: [1,95]
numRadiomics = 95
keyName = "hypertension_bp_plus_history$"
srcSuffix = "_Volume_s15_95radiomics.npy"
class01Percent = [0.4421085660495764, 0.5578914339504236]  # according to HBP tag and xml image.

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

    # make sure volume ID and volume path has strict corresponding order
    volumePaths = []  # number of volumes is about 2 times of IDList
    IDsCorrespondVolumes = []

    volumePathsFile = os.path.join(dataDir, mode + f"_{ODOS}_FeatureSelection_VolumePaths.txt")
    IDsCorrespondVolumesPathFile = os.path.join(dataDir, mode + f"_{ODOS}_FeatureSelection_IDsCorrespondVolumes.txt")

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
            resultList = fnmatch.filter(allVolumesList, "*/" + ID + f"_O[D,S]_*{srcSuffix}")  # for OD or OS data
            resultList.sort()
            numVolumes = len(resultList)
            if 0 == numVolumes:
                nonexistIDList.append(ID)
            else:
                volumePaths += resultList
                IDsCorrespondVolumes += [ID, ] * numVolumes  # multiple IDs

        if len(nonexistIDList) > 0:
            print(f"nonExistIDList of {ODOS} in {mode}:\n {nonexistIDList}")

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
    volumes = np.empty((NVolumes, numRadiomics), dtype=np.float)  # size:NxnRadiomics
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(np.float)
        volumes[i, :] = oneVolume[0, :]  # as oneVolume has a size [1, nRadiomics]

    fullLabels = readBESClinicalCsv(clinicalGTPath)

    # get HBP labels
    labelTable = np.empty((NVolumes, 2), dtype=np.float)  # size: Nx2 with (ID, Hypertension)
    for i in range(NVolumes):
        id = int(IDsCorrespondVolumes[i])
        labelTable[i, 0] = id
        labelTable[i, 1] = fullLabels[id][keyName]

    # save log information:
    print(f"{mode} dataset feature selection: NVolumes={NVolumes}\n")
    rate1 = labelTable[:, 1].sum() * 1.0 / NVolumes
    rate0 = 1 - rate1
    print(f"{mode} dataset feature selection: proportion of 0,1 = [{rate0},{rate1}]\n")

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

    # load training data, validation, and test data
    # volumes: volumes of all patient in this data: NxnRadiomics;
    # labelTable: numpy array with columns: [patientID, Hypertension(0/1)]
    trainVolumes, trainLabels = retrieveImageData_label("training")
    validationVolumes, validationLabels = retrieveImageData_label("validation")
    testVolumes, testLabels = retrieveImageData_label("test")

    # concatinate all data crossing training, validation, and test
    volumes = np.concatenate((trainVolumes, validationVolumes, testVolumes), axis=0)
    labels = np.concatenate((trainLabels, validationLabels, testLabels), axis=0)
    assert len(volumes) == len(labels)

    print("\n== Sequential backward feature selection w.r.t. HBP ==============")
    x = volumes  # only use radiomics features for sequential backward feature selection
    y = labels[:, 1]  # hypertension
    print(f"After concatenating training, validation and test data, it has {len(y)} samples.")

    # store the full feature names and its indexes in the x:
    fullFtrNames = []
    fullFtrIndexes = []
    for index in range(numRadiomics):
        fullFtrNames.append(radiomics95Features[index])
        fullFtrIndexes.append(index)
    print(f"Initial input features before feature selection:\n{fullFtrNames}")
    print("")

    # ================sequential backward feature selection========================
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
            nextIndexes = curIndexes[0:i] + curIndexes[i + 1:]
            nextClf = sm.Logit(y, x[:, tuple(nextIndexes)]).fit(disp=0)
            nextAIC = nextClf.aic
            if nextAIC < minAIC:
                minAIC = nextAIC
                minIndexes = nextIndexes
                minFtrs = curFtrs[0:i] + curFtrs[i + 1:]
                minClf = nextClf
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

    # ===Redo logistic regression with selected features===========
    clf = sm.Logit(y, x[:, tuple(curIndexes)]).fit(disp=0)
    print(clf.summary())
    predict = clf.predict(x[:, tuple(curIndexes)])
    accuracy = np.mean((predict >= 0.5).astype(np.int) == y)
    print(f"Accuracy of using {curFtrs} \n to predict hypertension with cutoff 0.5: {accuracy}")
    threhold_ACC_TPR_TNR_Sum = search_Threshold_Acc_TPR_TNR_Sum_WithProb(y, predict)
    print("With a different cut off with max(ACC+TPR+TNR):")
    print(threhold_ACC_TPR_TNR_Sum)
    print("Where:")
    for i in range(len(curFtrs)):
        print(f"x{i + 1} = {curFtrs[i]}")

    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"========== End of 95-radiomics Features Selection ============ ")


if __name__ == "__main__":
    main()
