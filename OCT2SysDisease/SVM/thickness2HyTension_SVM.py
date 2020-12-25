# Use SVM method to predict hypertension from thickness map

from sklearn import svm
import glob
import sys
import os
import fnmatch
import numpy as np
import yaml
sys.path.append("../..")
from framework.ConfigReader import ConfigReader
sys.path.append(".")
from OCT2SysD_Tools import readBESClinicalCsv

def printUsage(argv):
    print("============ OCT Thickness to Hypertension Using SVM =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_full_path")

def retrieveImageData_label(mode, hps):
    '''

    :param mode: "training", "validation", or "test"
    :param hps:
    :return: newVolumes: Nxfeatures
             targets: N of {-1,1}
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

    # normalize training volumes and save mean and std for using in validation and test data
    epsilon = 1.0e-8
    normalizationYamlPath = os.path.join(hps.netPath, hps.trainNormalizationStdMeanYamlName)
    if mode == "training":
        std = np.std(volumes)
        mean = np.mean(volumes)
        volumes = (volumes - mean) / (std + epsilon)  # size: NxCxHxW
        with open(normalizationYamlPath, 'w') as outfile:
            d = {"std": std.item(), "mean": mean.item()}
            yaml.dump(d, outfile, default_flow_style=False)

    elif (mode == "validation") or (mode == "test"):
        with open(normalizationYamlPath, 'r') as infile:
            cfg = yaml.load(infile, Loader=yaml.FullLoader)
        volumes = (volumes - cfg["mean"]) / (cfg["std"] + epsilon)  # size: NxCxHxW
    else:
        print(f"OCT2SysDiseaseDataSet mode error")
        assert False

    # get all target label {0, 1}
    fullLabels = readBESClinicalCsv(hps.GTPath)
    targets = np.empty((NVolumes,), dtype=np.int)
    for i, id in enumerate(IDsCorrespondVolumes):
        oneLabel = fullLabels[int(id)][hps.appKey]
        if "gender" == hps.appKey:
            oneLabel = oneLabel - 1
        #if oneLabel==0: # make sure label in {-1,1}
        #    oneLabel = -1
        targets[i] = oneLabel

    # flat feature plane
    newVolumes = volumes.reshape(NVolumes, -1)
    return newVolumes, targets


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


    # train SVM


    # test and print out



    # load
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)

    prediction =clf.predict([[2., 2.]])
    print(f"prediction: {prediction}")

if __name__ == "__main__":
    main()