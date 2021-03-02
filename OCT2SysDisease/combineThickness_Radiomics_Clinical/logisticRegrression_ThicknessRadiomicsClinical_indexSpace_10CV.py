# Sequential Backward feature selection on Thickness Radiomics Clinical in index space
'''
input: 100 3D radiomics feature in index space:
           "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s3tos8_100radiomics_indexSpace"
       9x9 thickness features:
           "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thickness9Sector_9x9"
       clinical data:
           10 clinical features ['Age', 'IOP', 'AxialLength', 'Pulse', 'Drink', 'Glucose', 'Triglyceride', 'BMI', 'WaistHipRate', 'LDLoverHDL']
           "/home/hxie1/data/BES_3K/GTs/BESClinicalGT_Analysis.csv"


10 fold crossing validation experiment:
output result in csv format:
CV   #TrainningSamples #ValidationSamples #TestSamples #TrainingAccuracy #ValidationAccuray #TestAccuracy

Algorithm:
1  use /home/hxie1/data/BES_3K/log/logisticReg_thicknessRadiomicsClinical_10CV/allIDwith10ClinicalFtrs.txt which filtered out non-existing clinical data ID
   and deleted myopia, glaucoma, macula diseases.
2  read al clinical data into a fullTable, get all ID's HBP Label;
3  according to HBP label, divide 1895 IDs into 10 folds.
4  for 10 CV test:
   A Assemble clinical features, thickness features, and radiomics features.
   B delete outliers.
   C logistic regression.
   D output result in csv format.
'''

radiomicsDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s3tos8_100radiomics_indexSpace"
thicknessDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thickness9Sector_9x9"
clinicalGTPath = "/home/hxie1/data/BES_3K/GTs/BESClinicalGT_Analysis.csv"
IDPath = "/home/hxie1/data/BES_3K/log/logisticReg_thicknessRadiomicsClinical_10CV/allIDwith10ClinicalFtrs.txt"
K = 10 # 10-fold Cross validation

outputDir = "/home/hxie1/data/BES_3K/log/logisticReg_thicknessRadiomicsClinical_10CV"

ODOS = "ODOS"

hintName= "3DThickRadioClinic_10CV"  # Thickness Radiomics Clinical

# input radiomics size: [1,numRadiomics]
numRadiomics = 100
numThickness = 81
numClinicalFtr=10
keyName = "hypertension_bp_plus_history$"
class01Percent = [0.449438202247191,0.550561797752809]
appKeys= ["hypertension_bp_plus_history$", "gender", "Age$", 'IOP$', 'AxialLength$', 'Height$', 'Weight$',
          'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$', 'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',
          'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015', 'TG$_Corrected2015']

inputClinicalFeatures= ['Age', 'IOP', 'AxialLength', 'Pulse', 'Drink', 'Glucose', 'Cholesterol', 'Triglyceride', 'BMI', 'LDLoverHDL']
clinicalFeatureColIndex= (3, 4, 5, 11, 12, 13, 15, 18, 19, 21)   # in label array index

output2File = True

import glob
import sys
import os
import fnmatch
import numpy as np
sys.path.append("../..")
from framework.measure import search_Threshold_Acc_TPR_TNR_Sum_WithProb
from OCT2SysD_Tools import readBESClinicalCsv

import datetime
import statsmodels.api as sm


def main():
    # prepare output file
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(outputDir, f"logisticRegression_{hintName}_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    print(f"===============Logistic Regression for {hintName} ================")

    # 1  use /home/hxie1/data/BES_3K/log/logisticReg_thicknessRadiomicsClinical_10CV/allIDwith10ClinicalFtrs.txt which filtered out non-existing clinical data ID
    #    and deleted ID of myopia, glaucoma, and macula diseases.
    with open(IDPath, 'r') as idFile:
        allIDList = idFile.readlines()
    allIDList = [item[0:-1] for item in allIDList]  # erase '\n'

    # 2  read al clinical data into a fullTable, get all ID's HBP Label;
    # read clinical GT, and extract hypertension with history tag
    fullLabels = readBESClinicalCsv(clinicalGTPath)
    nHBP0 = 0  # statistics hypertension
    nHBP1 = 0
    for ID in allIDList:
        if fullLabels[ID][keyName] == 1:
            nHBP1 += 1
        elif fullLabels[ID][keyName] == 0:
            nHBP0 += 1
        else:
            continue
    nHBP01 = nHBP0 + nHBP1
    print(f"in IDFile: {IDPath}: {len(allIDList)} patients, taggedHBP0 = {nHBP0}, taggedHBP1 = {nHBP1}, total {nHBP01} patients.")

    ID_HBP_Array = np.zeros((nHBP01, 2), dtype=np.uint32)  # ID and HBP
    i = 0
    for ID in allIDList:
        tag = fullLabels[ID][keyName]
        if 1 == tag or 0 == tag:
            ID_HBP_Array[i, 0] = int(ID)
            ID_HBP_Array[i, 1] = int(tag)
            i += 1

    # 3  according to HBP label, divide 1895 IDs into 10 folds.
    # K-Fold division
    patientID_0 = ID_HBP_Array[np.nonzero(ID_HBP_Array[:, 1] == 0), 0]  # size: 1xnHBP0
    patientID_1 = ID_HBP_Array[np.nonzero(ID_HBP_Array[:, 1] == 1), 0]
    patientID_0 = list(set(list(patientID_0[0, :])))  # erase repeated IDs
    patientID_1 = list(set(list(patientID_1[0, :])))
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
        patientsSubList.append(patientID0SubList[i] + patientID1SubList[K - 1 - i])  # help each fold has same number of samples

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)  # recursive dir creation

    outputValidation = True
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
        with open(os.path.join(outputDir, f"testID_10Clinical_NoMGMDiseases_{K}CV_{k}.csv"), "w") as file:
            for v in partitions["test"]:
                file.write(f"{int(v)}\n")

        if outputValidation:
            with open(os.path.join(outputDir, f"validationID_10Clinical_NoMGMDiseases_{K}CV_{k}.csv"), "w") as file:
                for v in partitions["validation"]:
                    file.write(f"{int(v)}\n")

        with open(os.path.join(outputDir, f"trainID_10Clinical_NoMGMDiseases_{K}CV_{k}.csv"), "w") as file:
            for v in partitions["training"]:
                file.write(f"{int(v)}\n")

        print(f"CV: {k}/{K}: test: {len(partitions['test'])} patients;  validation: {len(partitions['validation'])} patients;  training: {len(partitions['training'])} patients;")

    # 4  for 10 CV test:
    #    4.1 Assemble clinical features, thickness features, and radiomics features.

    #    4.2 delete outliers.


    #    4.3 logistic regression.


    #    4.4 output result in csv format.





    # debug
    # allIDList = allIDList[0:2000:10]
    # print(f"choose a small ID set for debug: {len(allIDList)}")

    NVolumes = len(allIDList)
    print(f"From /home/hxie1/data/BES_3K/GTs/radiomics_ODOS_10CV, extract total {NVolumes} IDs.")

    # 2  read all clinical data into a fullTable-> labelTable;
    fullLabels = readBESClinicalCsv(clinicalGTPath)

    labelTable = np.empty((NVolumes, 22), dtype=np.float)  # size: Nx22
    # labelTable head: patientID,                                          (0)
    #             "hypertension_bp_plus_history$", "gender", "Age$",'IOP$', 'AxialLength$', 'Height$', 'Weight$', 'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$',
    # columnIndex:         1                           2        3       4          5             6          7             8              9                10
    #              'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',  'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015',
    # columnIndex:   11            12                           13                      14                       15                       16                  17
    #              'TG$_Corrected2015',  BMI,   WaistHipRate,  LDL/HDL
    # columnIndex:      18                 19       20         21
    for i in range(NVolumes):
        id = int(allIDList[i])
        labelTable[i, 0] = id

        # appKeys: ["hypertension_bp_plus_history$", "gender", "Age$", 'IOP$', 'AxialLength$', 'Height$', 'Weight$',
        #          'Waist_Circum$', 'Hip_Circum$', 'SmokePackYears$', 'Pulse$', 'Drink_quanti_includ0$', 'Glucose$_Corrected2015', 'CRPL$_Corrected2015',
        #          'Choles$_Corrected2015', 'HDL$_Corrected2015', 'LDL$_Correcetd2015', 'TG$_Corrected2015']
        for j, key in enumerate(appKeys):
            oneLabel = fullLabels[id][key]
            if "gender" == key:
                oneLabel = oneLabel - 1
            labelTable[i, 1 + j] = oneLabel

        # compute BMI, WaistHipRate, LDL/HDL
        if labelTable[i, 7] == -100 or labelTable[i, 6] == -100:
            labelTable[i, 19] = -100  # emtpty value
        else:
            labelTable[i, 19] = labelTable[i, 7] / (
                        (labelTable[i, 6] / 100.0) ** 2)  # weight is in kg, height is in cm.

        if labelTable[i, 8] == -100 or labelTable[i, 9] == -100:
            labelTable[i, 20] = -100
        else:
            labelTable[i, 20] = labelTable[i, 8] / labelTable[i, 9]  # both are in cm.

        if labelTable[i, 17] == -100 or labelTable[i, 16] == -100:
            labelTable[i, 21] = -100
        else:
            labelTable[i, 21] = labelTable[i, 17] / labelTable[
                i, 16]  # LDL/HDL, bigger means more risk to hypertension.

    # 3  filter IDs with existed 10 clinical features.
    #    save these ID for futher use.
    assert NVolumes == len(labelTable)
    IDList2 =[]  # this list guarantee all IDs have corresponding clinical features.
    for i,id in enumerate(allIDList):
        if -100 in labelTable[i, clinicalFeatureColIndex]:
            pass
        else:
            IDList2.append(id)
    NVolumes = len(IDList2)
    print(f"After check the existance of corresponding {numClinicalFtr} features, it remains {NVolumes} IDs")
    IDwith10ClinicalFtrPath = os.path.join(outputDir, "allIDwith10ClinicalFtrs.txt")
    with open(IDwith10ClinicalFtrPath, "w") as file:
        for v in IDList2:
            file.write(f"{v}\n")


    # 4  Assemble clinical features, thickness features, and radiomics features.
    ftrArray = np.zeros((NVolumes, numRadiomics+numThickness+numClinicalFtr), dtype=np.float)
    labels  = np.zeros((NVolumes, 2), dtype=np.int) # columns: id, HBP

    radiomicsVolumesList = glob.glob(radiomicsDir + f"/*_Volume_100radiomics.npy")
    for i,id in enumerate(IDList2):
        resultList = fnmatch.filter(radiomicsVolumesList, "*/" + id + f"_O[D,S]_*_Volume_100radiomics.npy")  # for OD or OS data
        resultList.sort()
        numVolumes = len(resultList)
        assert numVolumes > 0
        clinicalFtrs = labelTable[i,clinicalFeatureColIndex]
        HBPLabel = labelTable[i,1]
        for radioVolumePath in resultList:
            volumeName = os.path.basename(radioVolumePath)  # 330_OD_680_Volume_100radiomics.npy
            volumeName = volumeName[0:volumeName.find("_Volume_100radiomics.npy")]   # 330_OD_680

            thicknessPath = os.path.join(thicknessDir, f"{volumeName}_thickness9sector_9x9.npy") # dir + 330_OD_680_thickness9sector_9x9.npy
            ftrArray[i,0:numRadiomics] = np.load(radioVolumePath).flatten()
            ftrArray[i,numRadiomics: numRadiomics+numThickness] = np.load(thicknessPath).flatten()
            ftrArray[i,numRadiomics+numThickness: ] = clinicalFtrs
            labels[i,0] = id
            labels[i,1] = HBPLabel

    assert len(labels) == len(ftrArray)
    print(f"After assembling radiomics, thickness, and clinical features, total {len(labels)} records.")

    # 4.5  delete outliers:
    # normalize x in each feature dimension
    # normalization does not affect the data distribution
    # as some features with big values will lead Logit overflow.
    # But if there is outlier, normalization still lead overflow.
    # delete outliers whose z-score abs value > 3.
    outlierRowsList = []  # embedded outlier rows list, in which elements are tuples.
    nDeleteOutLierIteration = 0
    while True:
        x = ftrArray.copy()
        for outlierRows in outlierRowsList:
            x = np.delete(x, outlierRows, axis=0)
        N = len(x)
        xMean = np.mean(x, axis=0, keepdims=True)  # size: 1xnRadiomics+nThickness +nClinical
        xStd = np.std(x, axis=0, keepdims=True)

        xMean = np.tile(xMean, (N, 1))  # same with np.broadcast_to, or pytorch expand
        xStd = np.tile(xStd, (N, 1))
        xNorm = (x - xMean) / (xStd + 1.0e-8)  # Z-score

        newOutlierRows = tuple(set(list(np.nonzero(np.abs(xNorm) >= 3.0)[0].astype(np.int))))
        outlierRowsList.append(newOutlierRows)
        if deleteOutlierIterateOnce:  # general only once.
            break
        if len(newOutlierRows) == 0:
            break
        else:
            nDeleteOutLierIteration += 1

    print(f"Deleting outlier used {nDeleteOutLierIteration + 1} iterations.")
    outlierIDs = []
    remainIDs = labels[:, 0].copy()
    for outlierRows in outlierRowsList:
        outlierIDs = outlierIDs + list(remainIDs[outlierRows,])  # must use comma
        remainIDs = np.delete(remainIDs, outlierRows, axis=0)

    # print(f"ID of {len(outlierIDs)} outliers: \n {outlierIDs}")
    # print(f"ID of {len(remainIDs)} remaining IDs: \n {list(remainIDs)}")

    y = labels[:, 1].copy()  # hypertension
    x = ftrArray.copy()
    for outlierRows in outlierRowsList:
        y = np.delete(y, outlierRows, axis=0)
        x = np.delete(x, outlierRows, axis=0)

    # re-normalize x
    N = len(y)
    print(f"After deleting outliers, there remains {N} observations.")
    # use original mean and std before deleting outlier, otherwise there are always outliers with deleting once.
    xMean = np.mean(ftrArray, axis=0, keepdims=True)  # size: 1xnRadiomics+nThickness +nClinical
    xStd = np.std(ftrArray, axis=0, keepdims=True)
    #print(f"feature mean values for all data after deleting outliers: \n{xMean}")
    #print(f"feature std devs for all data after deleting outliers: \n{xStd}")
    xMean = np.tile(xMean, (N, 1))  # same with np.broadcast_to, or pytorch expand
    xStd = np.tile(xStd, (N, 1))
    x = (x - xMean) / (xStd + 1.0e-8)  # Z-score


    # 5  logistic regression.
    x = x.copy()
    y = y.copy()
    clf = sm.Logit(y, x).fit(maxiter=200, method="bfgs", disp=0)
    #clf = sm.GLM(y, x[:, tuple(curIndexes)], family=sm.families.Binomial()).fit(maxiter=135, disp=0)
    print(clf.summary())
    predict = clf.predict(x)
    accuracy = np.mean((predict >= 0.5).astype(np.int) == y)
    print (f"\n==================================================")
    print(f"Accuracy of using {hintName} \n to predict hypertension with cutoff 0.5: {accuracy}")
    threhold_ACC_TPR_TNR_Sum = search_Threshold_Acc_TPR_TNR_Sum_WithProb(y, predict)
    print("With a different cut off with max(ACC+TPR+TNR):")
    print(threhold_ACC_TPR_TNR_Sum)

    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"========== End of Logistic Regression of {hintName} ============ ")


if __name__ == "__main__":
    main()
