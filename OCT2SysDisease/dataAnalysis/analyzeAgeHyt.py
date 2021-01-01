# analyze average texture of each layer with hypertension, age, gender


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
import matplotlib.pyplot as plt

def printUsage(argv):
    print("============ Anaylze OCT Age relation with hypertension =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_full_path")

def retrieveImageData_label(mode, hps):
    '''

    :param mode: "training", "validation", or "test"
    :param hps:
    :return: a numpy array with columns: Nx13
             patientID, thickness0,thickness1,... thickness8, Hypertension(0/1), Age(value), gender(0/1);
             where thicknessx is the average thickness in each layer of size 31x25

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
    volumes = np.empty((NVolumes, hps.inputChannels, hps.imageH, hps.imageW), dtype=np.float) # size:NxCxHxW
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(np.float)
        volumes[i,] = oneVolume

    fullLabels = readBESClinicalCsv(hps.GTPath)

    observationTable = np.empty((NVolumes, 22), dtype=np.float) #  size: Nx22
    # table head: patientID,                                          (0)
    #             thicknessMean0,thicknessMean1,... thicknessMean8,   (1:10)
    #             thicknessStd0,thicknessStd1,... thicknessStd8,      (10:19)
    #             Hypertension(0/1), Age(value), gender(0/1);         (19:22)
    for i in range(NVolumes):
        id = int(IDsCorrespondVolumes[i])
        observationTable[i,0] = id

        oneVolume = volumes[i,]  # size:9xHxW
        thicknessMean = np.mean(oneVolume,axis=(1,2))
        thicknessStd = np.std(oneVolume, axis=(1, 2))
        observationTable[i,1:10] = thicknessMean
        observationTable[i, 10:19] = thicknessStd


        # appKeys: ["hypertension_bp_plus_history$", "Age$", "gender"]
        for j,key in enumerate(hps.appKeys):
            oneLabel = fullLabels[id][key]
            if "gender" == key:
                oneLabel = oneLabel - 1
            observationTable[i, 19+j] = oneLabel

    return observationTable


def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

        # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")

    # load thickness data and ground truth
    # load training data, validation, and test data
    # table head: patientID,                                          (0)
    #             thicknessMean0,thicknessMean1,... thicknessMean8,   (1:10)
    #             thicknessStd0,thicknessStd1,... thicknessStd8,      (10:19)
    #             Hypertension(0/1), Age(value), gender(0/1);         (19:22)
    trainObsv = retrieveImageData_label("training", hps)
    validationObsv = retrieveImageData_label("validation", hps)
    testObsv = retrieveImageData_label("test", hps)

    # divide into hypertension 0,1 to analyze
    trainObsv_hyt0 = trainObsv[np.nonzero(trainObsv[:,19]  == 0)]
    trainObsv_hyt1 = trainObsv[np.nonzero(trainObsv[:, 19] == 1)]
    validationObsv_hyt0 = validationObsv[np.nonzero(validationObsv[:, 19] == 0)]
    validationObsv_hyt1 = validationObsv[np.nonzero(validationObsv[:, 19] == 1)]
    testObsv_hyt0 = testObsv[np.nonzero(testObsv[:, 19] == 0)]
    testObsv_hyt1 = testObsv[np.nonzero(testObsv[:, 19] == 1)]

    # draw figure for each surface, and each data subset
    dataList = [
        [trainObsv_hyt0, trainObsv_hyt1, hps.target+"_train"],
        [validationObsv_hyt0, validationObsv_hyt1, hps.target+"_validation" ],
        [testObsv_hyt0, testObsv_hyt1, hps.target+"_test"],
    ]

    x = [0, 1, 2] # indicate training, validation andd test set
    hyt0_ageMean = [-1,] *3
    hyt1_ageMean = [-1, ] * 3
    hyt0_ageStd  = [-1,] *3
    hyt1_ageStd = [-1, ] * 3
    pValueAge = [-1,]*3
    datasetName = ["Training", "Validation", "Test"]

    for i, dataSet in enumerate(dataList):
        hyt0_ageMean[i] = np.mean(dataSet[0][:, 20], axis=0)
        hyt0_ageStd[i] = np.std(dataSet[0][:, 20], axis=0)
        hyt1_ageMean[i] = np.mean(dataSet[1][:, 20], axis=0)
        hyt1_ageStd[i] = np.std(dataSet[1][:, 20], axis=0)
        _, pValueAge[i] = stats.ttest_ind(dataSet[0][:, 20], dataSet[1][:, 20])

    figureName = "Age_Hypertension_in3dataset.png"
    fig = plt.figure()
    plt.errorbar(x, hyt0_ageMean, yerr=hyt0_ageStd, label='NoHypertension', capsize=3)
    plt.errorbar(x, hyt1_ageMean, yerr=hyt1_ageStd, label='Hypertension', capsize=3)

    plt.xlabel("dataSet")
    plt.ylabel("Age mean/std (year)")
    plt.legend(loc='upper right')

    for i in x:
        plt.annotate(datasetName[i], (x[i], 60))

    outputFilePath = os.path.join(hps.outputDir, figureName)
    plt.savefig(outputFilePath)
    plt.close()

    figureName = "Pvalue_tTest_Age_Hyt.png"
    fig = plt.figure()

    plt.scatter(x, pValueAge)
    ylocation = sum(pValueAge)/3.0
    for i in range(3):
        txt = f"{pValueAge[i]:.3f}"
        plt.annotate(txt, (x[i], pValueAge[i]))
        plt.annotate(datasetName[i], (x[i], ylocation))

    plt.xlabel("dataset")
    plt.ylabel("PValue of Hypertension and Nohypertension")

    outputFilePath = os.path.join(hps.outputDir, figureName)
    plt.savefig(outputFilePath)
    plt.close()

if __name__ == "__main__":
    main()
    print(f"================End of anlayzing Age and Hypertension===============")