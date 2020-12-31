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
    print("============ Anaylze OCT Texture map relation with hypertension =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_full_path")

def retrieveImageData_label(mode, hps):
    '''

    :param mode: "training", "validation", or "test"
    :param hps:
    :return: a numpy array with columns: Nx13
             patientID, texture0,texture1,... texture8, Hypertension(0/1), Age(value), gender(0/1);
             where texturex is the average texture in each layer of size 31x25

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
    #             textureMean0,textureMean1,... textureMean8,   (1:10)
    #             textureStd0,textureStd1,... textureStd8,      (10:19)
    #             Hypertension(0/1), Age(value), gender(0/1);         (19:22)
    for i in range(NVolumes):
        id = int(IDsCorrespondVolumes[i])
        observationTable[i,0] = id

        oneVolume = volumes[i,]  # size:9xHxW
        textureMean = np.mean(oneVolume,axis=(1,2))
        textureStd = np.std(oneVolume, axis=(1, 2))
        observationTable[i,1:10] = textureMean
        observationTable[i, 10:19] = textureStd


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

    # load texture data and ground truth
    # load training data, validation, and test data
    # table head: patientID,                                          (0)
    #             textureMean0,textureMean1,... textureMean8,   (1:10)
    #             textureStd0,textureStd1,... textureStd8,      (10:19)
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

    nLayers = 9
    x = np.arange(nLayers)

    for dataSet in dataList:
        figureName = dataSet[2]+"_texture_layers.png"
        fig = plt.figure()

        hyt0_mean = np.mean(dataSet[0][:,1:10], axis=0)
        hyt0_std = np.std(dataSet[0][:, 1:10], axis=0)   # this std in sample dimension
        #hyt0_std  = np.mean(dataSet[0][:,10:19], axis=0)  # this std in channle plan
        hyt1_mean = np.mean(dataSet[1][:, 1:10], axis=0)
        hyt1_std = np.std(dataSet[1][:, 1:10], axis=0)
        #hyt1_std  = np.mean(dataSet[1][:, 10:19], axis=0)

        plt.errorbar(x, hyt0_mean, yerr=hyt0_std, label='no hypertension', capsize=3)
        plt.errorbar(x, hyt1_mean, yerr=hyt1_std, label='hypertension', capsize=3)

        plt.xlabel("Layer")
        plt.ylabel("Mean/std of texture intensity")
        plt.legend(loc='upper right')

        outputFilePath = os.path.join(hps.outputDir, figureName)
        plt.savefig(outputFilePath)
        plt.close()

    for dataSet in dataList:
        figureName = dataSet[2] + "_Pvalue_t_textureMean.png"
        pValues = [-1]*nLayers # pValue is prob >=0
        fig = plt.figure()

        for i in range(nLayers):
            _, pValues[i] = stats.ttest_ind(dataSet[0][:,i+1], dataSet[1][:,i+1])


        plt.scatter(x, pValues)
        for i in range(nLayers):
            txt = f"{pValues[i]:.3f}"
            plt.annotate(txt, (x[i], pValues[i]))

        plt.xlabel("Layer")
        plt.ylabel("PValue of Hypertension and Nohypertension")

        outputFilePath = os.path.join(hps.outputDir, figureName)
        plt.savefig(outputFilePath)
        plt.close()


    # chisqure need  observation and expectation has same length.
    '''
    
    for dataSet in dataList:
        figureName = dataSet[2] + "_Pvalue_chisquare_textureStdev.png"
        pValues = [-1]*nLayers # pValue is prob >=0
        fig = plt.figure()

        for i in range(nLayers):
            data0 = dataSet[0][:,i+10]
            data1 = dataSet[1][:,i+10]
            N0 = data0.size
            N1 = data1.size
            equalN = min(N0, N1)
            data0 = data0[0:equalN]**2  # chisqaure use variance
            data1 = data1[0:equalN]**2
            _, pValues[i] = stats.chisquare(data0, data1)


        plt.scatter(x, pValues)
        for i in range(nLayers):
            txt = f"{pValues[i]:.3f}"
            plt.annotate(txt, (x[i], pValues[i]))

        plt.xlabel("Layer")
        plt.ylabel("PValue of Hypertension and Nohypertension")

        outputFilePath = os.path.join(hps.outputDir, figureName)
        plt.savefig(outputFilePath)
        plt.close()
    '''







if __name__ == "__main__":
    main()
    print(f"================End of anlayzing Texture===============")