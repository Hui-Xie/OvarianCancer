# Use SVM method to predict hypertension from the 5th thickness and 6th texture map

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
    print("============ OCT Thickness and texture to Binary SysDisease Using SVM =============")
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
    volumes = np.empty((NVolumes, hps.inputChannels, hps.imageH, hps.imageW), dtype=float) # size:NxCxHxW
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(float)
        volumes[i,] = oneVolume

    # for texture and thickness enface map
    # normalize training volumes and save mean and std for using in validation and test data
    # normalization along each C (layer) dimension, and save as torch.pt file.
    epsilon = 1.0e-8
    normalizationFilePath_std = os.path.join(hps.netPath, hps.trainNormalizationStdMeanFileName + "_std.npy")
    normalizationFilePath_mean = os.path.join(hps.netPath, hps.trainNormalizationStdMeanFileName + "_mean.npy")
    if mode == "training":
        if (not os.path.isfile(normalizationFilePath_std)) or (not os.path.isfile(normalizationFilePath_mean)):
            std = np.std(volumes, axis=(0, 2, 3), keepdims=True)
            mean = np.mean(volumes, axis=(0, 2, 3), keepdims=True)
            np.save(normalizationFilePath_std, std)
            np.save(normalizationFilePath_mean, mean)
        else:
            std = np.load(normalizationFilePath_std)  # size: 1xCx1x1
            mean = np.load(normalizationFilePath_mean)

        std = np.broadcast_to(std, volumes.shape)  # equivalent with torch.expand_as
        mean = np.broadcast_to(mean, volumes.shape)
        volumes = (volumes - mean) / (std + epsilon)  # size: NxCxHxW

    elif (mode == "validation") or (mode == "test"):
        std = np.load(normalizationFilePath_std)  # size: 1xCx1x1
        mean = np.load(normalizationFilePath_mean)
        std = np.broadcast_to(std, volumes.shape)
        mean = np.broadcast_to(mean, volumes.shape)
        volumes = (volumes - mean) / (std + epsilon)  # size: NxCxHxW
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

    print("Program needs about 5-10 minutes according different input sizes, please wait...")

    # load training data, validation, and test data
    trainImages, trainTargets = retrieveImageData_label("training", hps)
    validationImages, validationTargets = retrieveImageData_label("validation", hps)
    testImages, testTargets = retrieveImageData_label("test", hps)

    print(f"finished data and label retrieving ......")

    # train SVM
    kernelList=("linear", "poly", "rbf", "sigmoid")
    nMethods = len(kernelList)
    trainAccList = [-1,]*nMethods
    validationAccList = [-1,]*nMethods
    testAccList = [-1,]*nMethods

    for i,kernel in enumerate(kernelList):
        classifier = svm.SVC(kernel=kernel)
        classifier.fit(trainImages, trainTargets)

        trainAccList[i] = classifier.score(trainImages, trainTargets)
        validationAccList[i] = classifier.score(validationImages, validationTargets)
        testAccList[i] = classifier.score(testImages, testTargets)
        print(f"finished {kernel} kernel fit and score......")

    # print result:
    print(f"\n=== Accuracies of the 5thThickness and 6thTexture to {hps.appKey} Prediction with different SVM kernels ====")
    print(f"Experiment: {hps.experimentName}")
    print(f"inputDataDir: {hps.dataDir}")
    print("========================================================================================================")
    print(f"SVM_Kernel,\t {','.join(str(x) for x in kernelList)}")
    print(f"Training,  \t {','.join(str(x) for x in trainAccList)}")
    print(f"Validation,\t {','.join(str(x) for x in validationAccList)}")
    print(f"Test,      \t {','.join(str(x) for x in testAccList)}")
    print("========================================================================================================")


if __name__ == "__main__":
    main()