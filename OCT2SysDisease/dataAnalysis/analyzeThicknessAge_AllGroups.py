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
    print("============ Anaylze OCT Thickness map relation with hypertension =============")
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
    volumes = np.empty((NVolumes, hps.inputChannels, hps.imageH, hps.imageW), dtype=float) # size:NxCxHxW
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(float)
        volumes[i,] = oneVolume

    fullLabels = readBESClinicalCsv(hps.GTPath)

    observationTable = np.empty((NVolumes, 22), dtype=float) #  size: Nx22
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

    allObsv = np.concatenate((trainObsv,validationObsv, testObsv), axis=0)

    nLayers = 9

    # divide age subgroup in hyt0 and hyt1:
    # Age subgroup: 50-55, 55-60, 60-65, 65-70, 70-80,80-95; total 6 groups

    ageDelimiters = [50, 55, 60, 65, 70, 80, 95,]
    nAgeSections = len(ageDelimiters)-1
    all_AgeSubGroups = []
    all_AgeSubGroups_mean = np.empty((nAgeSections, nLayers),dtype=float)
    all_AgeSubGroups_std = np.empty((nAgeSections, nLayers),dtype=float)

    for i in range(nAgeSections):
        all_AgeSubGroups.append(allObsv[np.nonzero((ageDelimiters[i] <= allObsv[:,20]) & (allObsv[:,20]< ageDelimiters[i+1]))])

    # print statistics information for different age group.

    print("\n=============================================================")
    print(f"======================All patients============================")
    print(f"AgeRange:\t", end="")
    for i in range(nAgeSections):
        print(f"{ageDelimiters[i]}-{ageDelimiters[i+1]},\t", end="")
    print("")

    print("All:\t", end="")
    for i in range(nAgeSections):
        print(f"{len(all_AgeSubGroups[i])},\t", end="")
    print("")
    print("=============================================================")

    for i in range(nAgeSections):
        all_AgeSubGroups_mean[i,] = np.mean(all_AgeSubGroups[i][:,1:10], axis=0)
        all_AgeSubGroups_std[i,] =  np.std(all_AgeSubGroups[i][:, 1:10], axis=0)

    for i in range(nLayers):
        figureName = "All"+f"_thickness_layer{i}.png"
        fig = plt.figure()

        plt.errorbar(ageDelimiters[0:-1], all_AgeSubGroups_mean[:,i], yerr=all_AgeSubGroups_std[:,i], label=f'allPatients_Layer{i}', capsize=3)

        plt.xlabel("Age")
        plt.ylabel("Mean/std thickness (micrometer)")
        plt.legend(loc='upper right')

        outputFilePath = os.path.join(hps.outputDir, figureName)
        plt.savefig(outputFilePath)
        plt.close()

if __name__ == "__main__":
    main()
    print(f"================End of anlayzing thickness===============")