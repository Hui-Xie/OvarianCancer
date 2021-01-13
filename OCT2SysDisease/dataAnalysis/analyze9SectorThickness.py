# analyze 9-sector thickness change over some risk factors.
# np.polyfit to draw regression line.


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
    volumes = np.empty((NVolumes, hps.imageH), dtype=np.float) # size:NxH for 9 sector array
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(np.float).squeeze(axis=-1)
        volumes[i,:] = oneVolume

    fullLabels = readBESClinicalCsv(hps.GTPath)

    labelTable = np.empty((NVolumes, 13), dtype=np.float) #  size: Nx11
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

    # concatinate all data crossing training, validation, and test
    volumes = np.concatenate((trainVolumes, validationVolumes, testVolumes), axis=0)
    labels = np.concatenate((trainLabels, validationLabels, testLabels), axis=0)

    nSectors = hps.imageH
    continuousAppKeys = ["Age",'IOP', 'AxialLength','SmokePackYears', "BMI", "WaistHipRate",]
    continuousAppKeyColIndex = [3,4,5,10,11,12]
    binaryAppKeys = ["hypertension_bp_plus_history", "gender",]
    binaryAppKeyColIndex = [1,2]
    layerName= "5thThickness"

    # draw continuous app keys lines.
    for sectorIndex in range(nSectors):
        for (keyIndex, colIndex) in enumerate(continuousAppKeyColIndex):
            figureName = f"sector{sectorIndex}_{continuousAppKeys[keyIndex]}_{layerName}"
            fig = plt.figure()

            x = labels[:,colIndex]
            y = volumes[:,sectorIndex]

            # delete the empty value of "-100"
            emptyRows = np.nonzero(x == -100)
            if continuousAppKeys[keyIndex] == "IOP":
                extraEmptyRows = np.nonzero(x == 99)
                emptyRows = emptyRows + extraEmptyRows
            x = np.delete(x, emptyRows, 0)
            y = np.delete(y, emptyRows, 0)
            print(f"{figureName}: deleted IDs:\n{labels[emptyRows,0]}\n")

            plt.scatter(x, y)
            m, b = np.polyfit(x,y, 1)
            plt.plot(x, m * x + b, 'r-')
            plt.xlabel(continuousAppKeys[keyIndex])
            plt.ylabel(f"Thickness_Sector{sectorIndex} (μm)")

            outputFilePath = os.path.join(hps.outputDir, figureName + ".png")
            plt.savefig(outputFilePath)
            plt.close()


    print(f"================ End of anlayzing 9-sector thickness  ===============")

if __name__ == "__main__":
    main()
