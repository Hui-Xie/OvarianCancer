# analyze each pixel t-test crossing training data, validatiion and test data set.
# for 9x15x12 input image.



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
    print("============ Anaylze OCT Thickness or texture map relation with hypertension =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_full_path")

def retrieveImageData_label(mode, hps):
    '''

    :param mode: "training", "validation", or "test"
    :param hps:
    :return: volumes: volumes of all patient in this data: NxCxHxW;
             labelTable: numpy array with columns: patientID, Hypertension(0/1), Age(value), gender(0/1);

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

    labelTable = np.empty((NVolumes, 4), dtype=np.float) #  size: Nx4
    # table head: patientID,                                          (0)
    #             Hypertension(0/1), Age(value), gender(0/1);         (1:4)
    for i in range(NVolumes):
        id = int(IDsCorrespondVolumes[i])
        labelTable[i,0] = id

        # appKeys: ["hypertension_bp_plus_history$", "Age$", "gender"]
        for j,key in enumerate(hps.appKeys):
            oneLabel = fullLabels[id][key]
            if "gender" == key:
                oneLabel = oneLabel - 1
            labelTable[i, 1+j] = oneLabel

    return volumes, labelTable


def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

        # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")

    # load training data, validation, and test data
    # volumes: volumes of all patient in this data: NxCxHxW;
    # labelTable: numpy array with columns: patientID, Hypertension(0/1), Age(value), gender(0/1);
    trainVolumes, trainLabels = retrieveImageData_label("training", hps)
    validationVolumes, validationLabels = retrieveImageData_label("validation", hps)
    testVolumes, testLabels = retrieveImageData_label("test", hps)

    # divide into hypertension 0,1 to analyze
    trainVolumes_hyt0 = trainVolumes[np.nonzero(trainLabels[:,1]  == 0)]
    trainVolumes_hyt1 = trainVolumes[np.nonzero(trainLabels[:, 1] == 1)]
    validationVolumes_hyt0 = validationVolumes[np.nonzero(validationLabels[:, 1] == 0)]
    validationVolumes_hyt1 = validationVolumes[np.nonzero(validationLabels[:, 1] == 1)]
    testVolumes_hyt0 = testVolumes[np.nonzero(testLabels[:, 1] == 0)]
    testVolumes_hyt1 = testVolumes[np.nonzero(testLabels[:, 1] == 1)]

    _, pValuesTrain = stats.ttest_ind(trainVolumes_hyt0, trainVolumes_hyt1, axis=0)
    _, pValuesValidation = stats.ttest_ind(validationVolumes_hyt0, validationVolumes_hyt1, axis=0)
    _, pValuesTest = stats.ttest_ind(testVolumes_hyt0, testVolumes_hyt1, axis=0)

    # mask[c,h,w]=1 means the t-test at location pvalue(c,h,w) < 0.05 crossing trianing, validation and test
    mask = (pValuesTrain <=0.05).astype(np.int) * (pValuesValidation <=0.05).astype(np.int) * (pValuesTest <=0.05).astype(np.int)
    C, H, W = mask.shape

    # simple statistic
    print(f"inputDataDir: {hps.dataDir}")
    print(f"mask shape: {mask.shape}")
    print(f"total statistical significance pixel number: {np.count_nonzero(mask)}")
    for c in range(C):
        print(f"at channel {c}, statistical significance pixel number: {np.count_nonzero(mask[c,])}")

    # save mask
    maskPath = os.path.join(hps.outputDir + f"mask_{C}x{H}x{W}.npy")
    np.save(maskPath, mask)

if __name__ == "__main__":
    main()
    print(f"================ End of anlayzing thickness in each pixel ===============")