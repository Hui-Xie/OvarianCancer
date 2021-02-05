# divide data into 10 fold, and save its 10-fold ID into GTs directory

import numpy as np
import torch
import os
import random

import glob
import fnmatch

import sys
sys.path.append(".")
from OCT2SysD_Tools import readBESClinicalCsv
import datetime

sys.path.append("../..")
from framework.ConfigReader import ConfigReader

output2File = True

def printUsage(argv):
    print("============ Divid data into Folds =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_full_path")


def main():
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

    trainIDPath = hps.trainingDataPath
    validationIDPath = hps.validationDataPath
    testIDPath = hps.testDataPath

    IDPathList=[trainIDPath, validationIDPath, testIDPath]

    IDList =[]
    for IDPath in IDPathList:
        with open(IDPath, 'r') as idFile:
            IDList += idFile.readlines()
    IDList = [item[0:-1] for item in IDList]  # erase '\n'

    # get all correct volume numpy path
    allVolumesList = glob.glob(hps.dataDir + f"/*{hps.volumeSuffix}")
    nonexistIDList = []
    multipleImages_IDList=[]

    # make sure volume ID and volume path has strict corresponding order
    volumePaths = []  # number of volumes is about 2 times of IDList
    IDsCorrespondVolumes = []

    volumePathsFile = os.path.join(hps.dataDir, "All"+ f"_{hps.ODOS}_VolumePaths.txt")
    IDsCorrespondVolumesPathFile = os.path.join(hps.dataDir, "All" + f"_{hps.ODOS}_IDsCorrespondVolumes.txt")

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
            resultList = fnmatch.filter(allVolumesList, "*/" + ID + f"_{hps.ODOS}_*{hps.volumeSuffix}")
            resultList.sort()
            numVolumes = len(resultList)
            if 0 == numVolumes:
                nonexistIDList.append(ID)
            elif numVolumes > 1:
                multipleImages_IDList.append(ID)
            else:
                volumePaths += resultList
                IDsCorrespondVolumes += [ID, ]  # one ID, one volume

        if len(nonexistIDList) > 0:
            print(f"nonExistIDList of {hps.ODOS} in all data sets:\n {nonexistIDList}")
        if len(multipleImages_IDList) > 0:
            print(f"List of ID corresponding multiple {hps.ODOS} images in all data sets:\n {multipleImages_IDList}")

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
    volumes = np.empty((NVolumes, hps.inputChannels, hps.imageH),  dtype=np.float)  # size:NxCxH for 9x9 sector array
    for i, volumePath in enumerate(volumePaths):
        oneVolume = np.load(volumePath).astype(np.float)
        volumes[i, :] = oneVolume
    volumes = volumes.reshape(-1, hps.inputChannels * hps.imageH * hps.imageW)  # size: Nx(CxHxW)

    # read clinical features
    fullLabels = readBESClinicalCsv(hps.GTPath)

    labelTable = np.empty((NVolumes, 22), dtype=np.float)  # size: Nx22
    # labelTable head: patientID,                                          (0)
    #             "hypertension_bp_plus_history$", "gender", "Age$",'IOP$', 'AxialLength$', 'Height$', 'Weight$', 'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$',
    # columnIndex:         1                           2        3       4          5             6          7             8              9                10
    #              'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',  'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015',
    # columnIndex:   11            12                           13                      14                       15                       16                  17
    #              'TG$_Corrected2015',  BMI,   WaistHipRate,  LDL/HDL
    # columnIndex:      18                 19       20         21
    for i in range(NVolumes):
        id = int(IDsCorrespondVolumes[i])
        labelTable[i, 0] = id

        # appKeys: ["hypertension_bp_plus_history$", "gender", "Age$", 'IOP$', 'AxialLength$', 'Height$', 'Weight$',
        #          'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$', 'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',
        #          'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015', 'TG$_Corrected2015']
        for j, key in enumerate(hps.appKeys):
            oneLabel = fullLabels[id][key]
            if "gender" == key:
                oneLabel = oneLabel - 1
            labelTable[i, 1 + j] = oneLabel

        # compute BMI, WaistHipRate, LDL/HDL
        if labelTable[i, 7] == -100 or labelTable[i, 6] == -100:
            labelTable[i, 19] = -100  # emtpty value
        else:
            labelTable[i, 19] = labelTable[i, 7] / ((labelTable[i, 6] / 100.0) ** 2)  # weight is in kg, height is in cm.

        if labelTable[i, 8] == -100 or labelTable[i, 9] == -100:
            labelTable[i, 20] = -100
        else:
            labelTable[i, 20] = labelTable[i, 8] / labelTable[i, 9]  # both are in cm.

        if labelTable[i, 17] == -100 or labelTable[i, 16] == -100:
            labelTable[i, 21] = -100
        else:
            labelTable[i, 21] = labelTable[i, 17] / labelTable[i, 16]  # LDL/HDL, bigger means more risk to hypertension.

    # concatenate selected thickness and clinical features, and then delete empty-feature patients
    inputClinicalFeatures = hps.inputClinicalFeatures
    clinicalFeatureColIndex = tuple(hps.clinicalFeatureColIndex)
    nClinicalFtr = len(clinicalFeatureColIndex)
    assert nClinicalFtr == hps.numClinicalFtr

    clinicalFtrs = labelTable[:, clinicalFeatureColIndex]
    # delete the empty value of "-100"
    emptyRows = np.nonzero(clinicalFtrs == -100)
    extraEmptyRows = np.nonzero(clinicalFtrs[:, inputClinicalFeatures.index("IOP")] == 99)  # missing IOP value
    emptyRows = (np.concatenate((emptyRows[0], extraEmptyRows[0]), axis=0),)

    inputThicknessFeatures = hps.inputThicknessFeatures
    thicknessFeatureColIndex = tuple(hps.thicknessFeatureColIndex)
    nThicknessFtr = len(thicknessFeatureColIndex)
    assert nThicknessFtr == hps.numThicknessFtr

    thicknessFtrs = volumes[:, thicknessFeatureColIndex]

    # concatenate sector thickness with multi variables:
    volumes = np.concatenate((thicknessFtrs, clinicalFtrs), axis=1)  # size: Nx(nThicknessFtr+nClinicalFtr)
    assert volumes.shape[1] == hps.inputWidth

    volumes = np.delete(volumes, emptyRows, 0)
    targetLabels = np.delete(labelTable, emptyRows, 0)[:, 1]  # for hypertension
    patientIDColumn = np.delete(labelTable, emptyRows, 0)[:, 0].astype(np.int)  # for patientID

    emptyRows = tuple(emptyRows[0])
    volumePaths = [path for index, path in enumerate(volumePaths) if index not in emptyRows]
    IDsCorrespondVolumes = [id for index, id in enumerate(IDsCorrespondVolumes) if index not in emptyRows]

    # update the number of volumes.
    NVolumes = len(volumes)

    print(f"data set including {hps.ODOS}: NVolumes={NVolumes}\n")
    rate1 = targetLabels.sum() * 1.0 / NVolumes
    rate0 = 1 - rate1
    print(f"data set including {hps.ODOS}: proportion of 0,1 = [{rate0},{rate1}]\n")

    # K-Fold division
    patientID_0 = patientIDColumn[np.nonzero(targetLabels==0)]
    patientID_1 = patientIDColumn[np.nonzero(targetLabels==1)]
    patientID_0 = list(set(list(patientID_0)))  # erase repeated IDs
    patientID_1 = list(set(list(patientID_1)))

    print(f"After erasing repeated ID: Num_response0 = {len(patientID_0)}, Num_response1= {len(patientID_1)}, total={len(patientID_0)+len(patientID_1)}")

    # split files in sublist, this is a better method than before.
    K = hps.K_Fold

    patientID0SubList = []
    step = len(patientID_0) // K
    for i in range(0, K * step, step):
        nexti = i + step
        patientID0SubList.append(patientID_0[i:nexti])
    for i in range(K * step, len(patientID_0)):
        patientID0SubList[i - K * step].append(patientID_0[i])

    patientID1SubList = []
    step = len(patientID_1) // K
    for i in range(0, K * step, step):
        nexti = i + step
        patientID1SubList.append(patientID_1[i:nexti])
    for i in range(K * step, len(patientID_1)):
        patientID1SubList[i - K * step].append(patientID_1[i])

    patientsSubList=[]
    for i in range(K):
        patientsSubList.append(patientID0SubList[i] + patientID1SubList[i])

    # partition for test, validation, and training
    outputValidation = hps.outputValidation

    if not os.path.exists(hps.outputCV_ID_Dir):
        os.makedirs(hps.outputCV_ID_Dir)  # recursive dir creation

    print("")
    for k in range(0, K):
        partitions = {}
        partitions["test"] = patientsSubList[k]

        if outputValidation:
            k1 = (k + 1) % K  # validation k
            partitions["validation"] = patientsSubList[k1]
        else:
            k1 = k
            partitions["validation"] = []

        partitions["training"] = []
        for i in range(K):
            if i != k and i != k1:
                partitions["training"] += patientsSubList[i]

        # save to file
        with open(os.path.join(hps.outputCV_ID_Dir, f"testID_{hps.inputWidth}Ftrs_{K}CV_{k}.csv"), "w") as file:
            for v in partitions["test"]:
                file.write(f"{int(v)}\n")

        if outputValidation:
            with open(os.path.join(hps.outputCV_ID_Dir, f"validationID_{hps.inputWidth}Ftrs_{K}CV_{k}.csv"), "w") as file:
                for v in partitions["validation"]:
                    file.write(f"{int(v)}\n")

        with open(os.path.join(hps.outputCV_ID_Dir, f"trainID_{hps.inputWidth}Ftrs_{K}CV_{k}.csv"), "w") as file:
            for v in partitions["training"]:
                file.write(f"{int(v)}\n")

        print(f"in CV: {k}/{K}: test: {len(partitions['test'])} patients;  validation: {len(partitions['validation'])} patients;  training: {len(partitions['training'])} patients;")


    if output2File:
        logOutput.close()
        sys.stdout = original_stdout


if __name__ == "__main__":
    main()

