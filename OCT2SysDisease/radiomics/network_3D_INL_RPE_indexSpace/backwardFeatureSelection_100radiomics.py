# Sequential Backward feature selection on 100 radiomics in index(pixel) space

dataDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s3tos8_100radiomics_indexSpace"
ODOS = "ODOS"
hintName= "3DIndexSpace"

radiomicsFeatures = [
"original_firstorder_10Percentile", # sample value :42.0
"original_firstorder_90Percentile", # sample value :122.0
"original_firstorder_Energy", # sample value :4546035246.0
"original_firstorder_Entropy", # sample value :2.290483429722902
"original_firstorder_InterquartileRange", # sample value :37.0
"original_firstorder_Kurtosis", # sample value :4.9019165966015015
"original_firstorder_Maximum", # sample value :255.0
"original_firstorder_Mean", # sample value :74.69587063807806
"original_firstorder_MeanAbsoluteDeviation", # sample value :25.09769550893551
"original_firstorder_Median", # sample value :66.0
"original_firstorder_Minimum", # sample value :0.0
"original_firstorder_Range", # sample value :255.0
"original_firstorder_RobustMeanAbsoluteDeviation", # sample value :16.297969041319497
"original_firstorder_RootMeanSquared", # sample value :81.6777083003706
"original_firstorder_Skewness", # sample value :1.309870242982822
"original_firstorder_TotalEnergy", # sample value :4546035246.0
"original_firstorder_Uniformity", # sample value :0.257001376438416
"original_firstorder_Variance", # sample value :1091.7749428199356
"original_glcm_Autocorrelation", # sample value :13.445557595443358
"original_glcm_ClusterProminence", # sample value :181.72086300382574
"original_glcm_ClusterShade", # sample value :19.446166987043277
"original_glcm_ClusterTendency", # sample value :6.382152775147923
"original_glcm_Contrast", # sample value :0.7569681474089494
"original_glcm_Correlation", # sample value :0.7877438390604021
"original_glcm_DifferenceAverage", # sample value :0.5321648546570841
"original_glcm_DifferenceEntropy", # sample value :1.310176093469075
"original_glcm_DifferenceVariance", # sample value :0.46133124978722395
"original_glcm_Id", # sample value :0.7648781309979223
"original_glcm_Idm", # sample value :0.7556187862889131
"original_glcm_Idmn", # sample value :0.9939580335140424
"original_glcm_Idn", # sample value :0.9570133845390073
"original_glcm_Imc1", # sample value :-0.3095395644942348
"original_glcm_Imc2", # sample value :0.8564980904211521
"original_glcm_InverseVariance", # sample value :0.39147550672037795
"original_glcm_JointAverage", # sample value :3.4697463969469755
"original_glcm_JointEnergy", # sample value :0.12217498988669677
"original_glcm_JointEntropy", # sample value :3.8295065172518408
"original_glcm_MaximumProbability", # sample value :0.270951894631309
"original_glcm_SumEntropy", # sample value :3.1120002276241197
"original_glcm_SumSquares", # sample value :1.7847802306392178
"original_gldm_DependenceEntropy", # sample value :6.678234307296107
"original_gldm_DependenceNonUniformity", # sample value :27541.935112123352
"original_gldm_DependenceNonUniformityNormalized", # sample value :0.0404174342046636
"original_gldm_DependenceVariance", # sample value :45.41983390219565
"original_gldm_GrayLevelNonUniformity", # sample value :175130.2469560649
"original_gldm_GrayLevelVariance", # sample value :1.838393018618642
"original_gldm_HighGrayLevelEmphasis", # sample value :14.131858117478211
"original_gldm_LargeDependenceEmphasis", # sample value :265.7640031286825
"original_gldm_LargeDependenceHighGrayLevelEmphasis", # sample value :2708.802916483842
"original_gldm_LargeDependenceLowGrayLevelEmphasis", # sample value :35.866339746892635
"original_gldm_LowGrayLevelEmphasis", # sample value :0.12124936233989302
"original_gldm_SmallDependenceEmphasis", # sample value :0.01945374983188274
"original_gldm_SmallDependenceHighGrayLevelEmphasis", # sample value :0.4641673242683492
"original_gldm_SmallDependenceLowGrayLevelEmphasis", # sample value :0.003015741865058172
"original_glrlm_GrayLevelNonUniformity", # sample value :64762.95411439012
"original_glrlm_GrayLevelNonUniformityNormalized", # sample value :0.2019612815507785
"original_glrlm_GrayLevelVariance", # sample value :2.4082273338471643
"original_glrlm_HighGrayLevelRunEmphasis", # sample value :17.34035339474268
"original_glrlm_LongRunEmphasis", # sample value :10.613395125558059
"original_glrlm_LongRunHighGrayLevelEmphasis", # sample value :111.31010096421248
"original_glrlm_LongRunLowGrayLevelEmphasis", # sample value :1.4575836342279298
"original_glrlm_LowGrayLevelRunEmphasis", # sample value :0.11217147029110729
"original_glrlm_RunEntropy", # sample value :4.554336370935227
"original_glrlm_RunLengthNonUniformity", # sample value :121398.84755314625
"original_glrlm_RunLengthNonUniformityNormalized", # sample value :0.3694146349109773
"original_glrlm_RunPercentage", # sample value :0.46753867759771467
"original_glrlm_RunVariance", # sample value :5.480309177473375
"original_glrlm_ShortRunEmphasis", # sample value :0.6232494732836416
"original_glrlm_ShortRunHighGrayLevelEmphasis", # sample value :12.140498420707608
"original_glrlm_ShortRunLowGrayLevelEmphasis", # sample value :0.06830135941130923
"original_glszm_GrayLevelNonUniformity", # sample value :963.3403279440431
"original_glszm_GrayLevelNonUniformityNormalized", # sample value :0.11046214057379235
"original_glszm_GrayLevelVariance", # sample value :7.043698800201288
"original_glszm_HighGrayLevelZoneEmphasis", # sample value :30.008026602453846
"original_glszm_LargeAreaEmphasis", # sample value :12944440.11271643
"original_glszm_LargeAreaHighGrayLevelEmphasis", # sample value :127297991.09402591
"original_glszm_LargeAreaLowGrayLevelEmphasis", # sample value :1600230.2391638118
"original_glszm_LowGrayLevelZoneEmphasis", # sample value :0.19626521041929643
"original_glszm_SizeZoneNonUniformity", # sample value :2113.1284256392614
"original_glszm_SizeZoneNonUniformityNormalized", # sample value :0.2423034543790003
"original_glszm_SmallAreaEmphasis", # sample value :0.49473352262254067
"original_glszm_SmallAreaHighGrayLevelEmphasis", # sample value :13.9363132320453
"original_glszm_SmallAreaLowGrayLevelEmphasis", # sample value :0.11258455448193426
"original_glszm_ZoneEntropy", # sample value :6.081747083200463
"original_glszm_ZonePercentage", # sample value :0.01279795490999168
"original_glszm_ZoneVariance", # sample value :12938334.646274097
"original_shape_Elongation", # sample value :0.1063750261465349
"original_shape_Flatness", # sample value :0.05904268066748336
"original_shape_LeastAxisLength", # sample value :34.183579998584754
"original_shape_MajorAxisLength", # sample value :578.9638887011225
"original_shape_Maximum2DDiameterColumn", # sample value :512.8781531709067
"original_shape_Maximum2DDiameterRow", # sample value :69.6419413859206
"original_shape_Maximum2DDiameterSlice", # sample value :519.339002964345
"original_shape_Maximum3DDiameter", # sample value :519.9278796140865
"original_shape_MeshVolume", # sample value :681129.2083333334
"original_shape_MinorAxisLength", # sample value :61.587298798481434
"original_shape_Sphericity", # sample value :0.42730924331409403
"original_shape_SurfaceArea", # sample value :87611.57880600162
"original_shape_SurfaceVolumeRatio", # sample value :0.1286269590763548
"original_shape_VoxelVolume", # sample value :681437.0
]

# for sequential backward feature choose, use all data at CV0
trainingDataPath = "/home/hxie1/data/BES_3K/GTs/radiomics_ODOS_10CV/trainID_segmented_10CV_0.csv"
validationDataPath = "/home/hxie1/data/BES_3K/GTs/radiomics_ODOS_10CV/validationID_segmented_10CV_0.csv"
testDataPath = "/home/hxie1/data/BES_3K/GTs/radiomics_ODOS_10CV/testID_segmented_10CV_0.csv"

clinicalGTPath = "/home/hxie1/data/BES_3K/GTs/BESClinicalGT_Analysis.csv"

outputDir = "/home/hxie1/data/BES_3K/GTs/radiomics_ODOS_10CV"

# input radiomics size: [1,numRadiomics]
numRadiomics = 100
keyName = "hypertension_bp_plus_history$"
srcSuffix = "_Volume_100radiomics.npy"
class01Percent = [0.449438202247191,0.550561797752809]  # according to HBP tag and xml image.

output2File = True
deleteOutlierIterateOnce = True


import glob
import sys
import os
import fnmatch
import numpy as np
sys.path.append("../../..")
from framework.measure import search_Threshold_Acc_TPR_TNR_Sum_WithProb
from OCT2SysD_Tools import readBESClinicalCsv

import datetime
import statsmodels.api as sm

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
    volumes = np.empty((NVolumes, numRadiomics), dtype=float)  # size:NxnRadiomics
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(float)
        volumes[i, :] = oneVolume[0, :]  # as oneVolume has a size [1, nRadiomics]

    fullLabels = readBESClinicalCsv(clinicalGTPath)

    # get HBP labels
    labelTable = np.empty((NVolumes, 2), dtype=np.int)  # size: Nx2 with (ID, Hypertension)
    for i in range(NVolumes):
        id = int(IDsCorrespondVolumes[i])
        labelTable[i, 0] = id
        labelTable[i, 1] = fullLabels[id][keyName]

    # save log information:
    print(f"{mode} dataset feature selection: NVolumes={NVolumes}")
    rate1 = labelTable[:, 1].sum() * 1.0 / NVolumes
    rate0 = 1 - rate1
    print(f"{mode} dataset {numRadiomics} feature selection: proportion of 0,1 = [{rate0},{rate1}]")

    return volumes, labelTable


def main():
    # prepare output file
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(outputDir, f"{numRadiomics}radiomics_{hintName}_FeatureSelection_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    print(f"===============Logistic Regression Feature Selection from {numRadiomics} radiomics ================")

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
    x = volumes.copy()  # only use radiomics features for sequential backward feature selection
    y = labels[:, 1].copy()  # hypertension
    N = len(x)  # number of samples.
    print(f"After concatenating training, validation and test data, it has {len(y)} samples.")

    # normalize x in each feature dimension
    # normalization does not affect the data distribution
    # as some features with big values will lead Logit overflow.
    # But if there is outlier, normalization still lead overflow.
    # delete outliers whose z-score abs value > 3.
    outlierRowsList= [] # embedded outlier rows list, in which elements are tuples.
    nDeleteOutLierIteration = 0
    while True:
        x = volumes.copy()
        for outlierRows in outlierRowsList:
            x = np.delete(x, outlierRows, axis=0)
        N = len(x)
        xMean = np.mean(x, axis=0, keepdims=True) # size: 1xnRadiomics
        xStd  = np.std(x, axis=0,  keepdims=True)

        xMean = np.tile(xMean, (N,1))  # same with np.broadcast_to, or pytorch expand
        xStd  = np.tile(xStd, (N,1))
        xNorm = (x -xMean)/(xStd+1.0e-8)  # Z-score

        newOutlierRows = tuple(set(list(np.nonzero(np.abs(xNorm)>= 3.0)[0].astype(np.int))))
        outlierRowsList.append(newOutlierRows)
        if deleteOutlierIterateOnce:  # general only once.
            break
        if len(newOutlierRows) ==0:
            break
        else:
            nDeleteOutLierIteration += 1

    print(f"Deleting outlier used {nDeleteOutLierIteration+1} iterations.")
    outlierIDs = []
    remainIDs = labels[:, 0].copy()
    for outlierRows in outlierRowsList:
        outlierIDs = outlierIDs +list(remainIDs[outlierRows,])  # must use comma
        remainIDs = np.delete(remainIDs, outlierRows, axis=0)

    print(f"ID of {len(outlierIDs)} outliers: \n {outlierIDs}")
    print(f"ID of {len(remainIDs)} remaining IDs: \n {list(remainIDs)}")

    y = labels[:, 1].copy()  # hypertension
    x = volumes.copy()
    for outlierRows in outlierRowsList:
        y = np.delete(y,outlierRows, axis=0)
        x = np.delete(x, outlierRows, axis=0)

    # re-normalize x
    N = len(y)
    # use original mean and std before deleting outlier, otherwise there are always outliers with deleting once.
    xMean = np.mean(volumes, axis=0, keepdims=True)  # size: 1xnRadiomics
    xStd = np.std(volumes, axis=0, keepdims=True)
    print(f"feature mean values for all data after deleting outliers: \n{xMean}")
    print(f"feature std devs for all data after deleting outliers: \n{xStd}")
    xMean = np.tile(xMean, (N, 1))  # same with np.broadcast_to, or pytorch expand
    xStd = np.tile(xStd, (N, 1))
    x = (x - xMean) / (xStd + 1.0e-8)  # Z-score

    # store the full feature names and its indexes in the x:
    fullFtrNames = []
    fullFtrIndexes = []
    for index in range(numRadiomics):
        fullFtrNames.append(radiomicsFeatures[index])
        fullFtrIndexes.append(index)
    print(f"Initial input features before feature selection:\n{fullFtrNames}")
    print("")

    # ================sequential backward feature selection========================
    print(f"============program is in sequential backward feature selection, please wait......==============")
    curIndexes = fullFtrIndexes.copy()
    curFtrs = fullFtrNames.copy()
    # use bfgs to avoid Singular matrix  # use ‘nm’ for Nelder-Mead to avoid Hessian failure
    curClf = sm.Logit(y, x[:, tuple(curIndexes)]).fit(maxiter=200, method="bfgs", disp=0)
    #curClf = sm.GLM(y, x[:, tuple(curIndexes)], family=sm.families.Binomial()).fit(maxiter=135, disp=0)
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
            nextClf = sm.Logit(y, x[:, tuple(nextIndexes)]).fit(maxiter=200, method="bfgs", disp=0)
            #nextClf = sm.GLM(y, x[:, tuple(nextIndexes)], family=sm.families.Binomial()).fit(maxiter=135, disp=0)
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
    print(f"The number of selected features = {len(curIndexes)}")
    print(f"selected features: {curFtrs}")
    print(f"selected feature indexes: {curIndexes}\n")


    # ===Redo logistic regression with selected features===========
    clf = sm.Logit(y, x[:, tuple(curIndexes)]).fit(maxiter=200, method="bfgs", disp=0)
    #clf = sm.GLM(y, x[:, tuple(curIndexes)], family=sm.families.Binomial()).fit(maxiter=135, disp=0)
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

    print(f"========== End of {numRadiomics}-radiomics Features Selection ============ ")


if __name__ == "__main__":
    main()
