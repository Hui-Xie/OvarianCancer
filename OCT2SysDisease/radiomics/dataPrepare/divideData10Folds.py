# divide all data into 10 folds
srcRadiomicsDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15_95radiomics"
clinicalGTPath = "/home/hxie1/data/BES_3K/GTs/BESClinicalGT_Analysis.csv"
outputDir = "/home/hxie1/data/BES_3K/GTs/radiomics_ODOS_10CV"

import numpy as np
import glob
import os
import fnmatch

import sys
sys.path.append("../network_95radiomics")
from OCT2SysD_Tools import readBESClinicalCsv
import datetime

output2File = True
keyName = "hypertension_bp_plus_history$"
srcSuffix = "_Volume_s15_95radiomics.npy"
K = 10   # K-fold cross validation.
outputValidation = True

def main():
    # prepare output file
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(outputDir, f"output_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    print(f"===============Divide IDs with 95 radiomics features and hypertension tags into 10 folds================")

    # read clinical GT, and extract hypertension with history tag
    fullLabels = readBESClinicalCsv(clinicalGTPath)
    nHBP0 = 0  # statistics hypertension
    nHBP1 = 0
    for ID in fullLabels:
        if fullLabels[ID][keyName] == 1:
            nHBP1 += 1
        elif fullLabels[ID][keyName] == 0:
            nHBP0 += 1
        else:
            continue
    nHBP01 = nHBP0 + nHBP1
    print(f"in the clinical GT files of {len(fullLabels)} patients, taggedHBP0 = {nHBP0}, taggedHBP1 = {nHBP1}, total {nHBP01} patients.")

    ID_HBP_Array = np.zeros((nHBP01, 2), dtype=np.uint32)
    i = 0
    for ID in fullLabels:
        tag = fullLabels[ID][keyName]
        if 1 == tag or 0 ==tag:
            ID_HBP_Array[i,0] = int(ID)
            ID_HBP_Array[i,1] = int(tag)
            i += 1

    # check ID with hypertension in srcRadiomicsDir, delete the not exist IDs
    allRadiomicsList = glob.glob(srcRadiomicsDir + f"/*{srcSuffix}")
    radimicsIDSet = set()
    for radiomicsFile in allRadiomicsList:
        filename = os.path.splitext(os.path.basename(radiomicsFile))[0]
        ID = filename[0: filename.find("_O")]
        if ID.isdigit():
            radimicsIDSet.add(int(ID))
    print(f"In radiomics dir, there are {len(radimicsIDSet)} unique IDs.")

    invalidIDRows = []
    # delete ID without its radiomic feature/segmented xml
    # delete MGM cases: high myopia(axilLength>=26), Glaucoma, Macula / Retina disease cases
    excludedKeys = ['Axiallength_26_ormore_exclude$', 'Glaucoma_exclude$', 'Retina_exclude$']
    for i in range(nHBP01):
        if not (ID_HBP_Array[i,0] in radimicsIDSet):
            invalidIDRows.append(i)
        for excludedKey in excludedKeys:
            if 1 == fullLabels[ID_HBP_Array[i,0]][excludedKey]:
                invalidIDRows.append(i)
                break
    print(f"delete IDs with high myopia, glaucoma and macula/retina disease")
    print(f"delete IDs without segmented files.")

    invalidIDRows = tuple(invalidIDRows)
    ID_HBP_Array = np.delete(ID_HBP_Array, invalidIDRows, 0)
    nHBP01 = len(ID_HBP_Array)
    nHBP1 = int(ID_HBP_Array[:,1].sum())
    nHBP0 = nHBP01- nHBP1
    print(f"After deleting invalid IDs in radiomics files, ID_HBP_Array remains HBP0={nHBP0}, HBP1={nHBP1}, total {nHBP01} patients.")
    print(f"valid HBP data set: proportion of 0,1 = [{nHBP0*1.0/nHBP01},{nHBP1*1.0/nHBP01}]")

    # divide all remaining IDs into K folds
    # K-Fold division
    patientID_0 = ID_HBP_Array[np.nonzero(ID_HBP_Array[:,1] == 0), 0] # size: 1xnHBP0
    patientID_1 = ID_HBP_Array[np.nonzero(ID_HBP_Array[:,1] == 1), 0]
    patientID_0 = list(set(list(patientID_0[0,:])))  # erase repeated IDs
    patientID_1 = list(set(list(patientID_1[0,:])))
    assert len(patientID_0) == nHBP0
    assert len(patientID_1) == nHBP1
    print(f"After erasing repeated ID: Num_response0 = {len(patientID_0)}, Num_response1= {len(patientID_1)}, total={len(patientID_0) + len(patientID_1)}")

    # split files in sublist, this is a better method than before.
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

    patientsSubList = []
    for i in range(K):
        patientsSubList.append(patientID0SubList[i] + patientID1SubList[K-1-i])   # help each fold has same number of samples


    if not os.path.exists(outputDir):
        os.makedirs(outputDir)  # recursive dir creation

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
        with open(os.path.join(outputDir, f"testID_segmented_{K}CV_{k}.csv"), "w") as file:
            for v in partitions["test"]:
                file.write(f"{int(v)}\n")

        if outputValidation:
            with open(os.path.join(outputDir, f"validationID_segmented_{K}CV_{k}.csv"),"w") as file:
                for v in partitions["validation"]:
                    file.write(f"{int(v)}\n")

        with open(os.path.join(outputDir, f"trainID_segmented_{K}CV_{k}.csv"), "w") as file:
            for v in partitions["training"]:
                file.write(f"{int(v)}\n")

        print(f"CV: {k}/{K}: test: {len(partitions['test'])} patients;  validation: {len(partitions['validation'])} patients;  training: {len(partitions['training'])} patients;")


    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"==========End of dividing IDs with segmented features and hypertension tag into 10 folds============ ")


if __name__ == "__main__":
    main()