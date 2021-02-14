# divide all data into 10 folds
srcRadiomicsDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15_95radiomics"
clinicalGTPath = "/home/hxie1/data/BES_3K/GTs/BESClinicalGT_Analysis.csv"
outputDir = "/home/hxie1/data/BES_3K/GTs/95radiomics_ODOS_10CV"

import numpy as np
import glob
import os
import fnmatch

import sys
sys.path.append(".")
from OCT2SysD_Tools import readBESClinicalCsv
import datetime

output2File = True
keyName = "hypertension_bp_plus_history$"
srcSuffix = "_Volume_s15_95radiomics.npy"

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

    nonexistIDRows = []
    for i in range(nHBP01):
        if not (ID_HBP_Array[i,0] in radimicsIDSet):
            nonexistIDRows.append(i)
    nonexistIDRows = tuple(nonexistIDRows)
    ID_HBP_Array = np.delete(ID_HBP_Array, nonexistIDRows, 0)
    nHBP01 = len(ID_HBP_Array)
    nHBP1 = ID_HBP_Array[:,1].sum()
    nHBP0 = nHBP01- nHBP1
    print(f"After deleting invalid IDs in radiomics files, ID_HBP_Array remains {nHBP01} patients: HBP0={nHBP0}, HBP1={nHBP1}.")

    # divide all remaining IDs into 10 folds


    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"==========End of dividing IDs with 95 radiomics features and hypertension tag into 10 folds============ ")


if __name__ == "__main__":
    main()