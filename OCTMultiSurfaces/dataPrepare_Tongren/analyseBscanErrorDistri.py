# Analyse the error distribution in each Bscan

import yaml
import glob
import os
import numpy as np
import json
from TongrenFileUtilities import *

yamlFilePath = "/home/hxie1/Projects/DeepLearningSeg/OCTMultiSurfaces/testConfig/expTongren_10Surfaces_allGoodBscans_20200512/allGoodBscans.yaml"

gtPath = "/local/vol00/scratch/Users/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet_10Surfaces_AllGoodBscans/test"
xmlPath = "/home/hxie1/data/OCT_Tongren/goodBscanPrediction_20200512/10SurfacesXml"
outputPath ="/home/hxie1/data/OCT_Tongren/goodBscanPrediction_20200512"
outputFile ="errorDistrInBscans.txt"

NumBscansPerPatient = 31
sIndex = 2  # specific surface index

# read good scan config
with open(yamlFilePath) as file:
    goodBscans = yaml.load(file, Loader=yaml.FullLoader)['goodBscans']

# read ground truth
IDList = glob.glob(gtPath + f"/patientID_CV?.json")
IDList.sort()
volumeNameList = []  #element: '/home/hxie1/data/OCT_Tongren/control/2044_OD_14191_Volume/20110527061539_OCT01.jpg'
for IDPath in IDList:
    with open(IDPath) as f:
        foldID = json.load(f)
    nSlices = len(foldID)
    for i in range(0,nSlices, NumBscansPerPatient):
        volumeNameList.append(foldID[str(i)])
volumeNameList =[os.path.basename(a[:a.rfind('/')]) for a in volumeNameList]  # element: 2044_OD_14191_Volume
patientIDList = [int(a[:a.find('_OD_')]) for a in volumeNameList]    # 2044

gtList = glob.glob(gtPath + f"/surfaces_CV?.npy")
gtList.sort()
gtSurfaces = None  # size: B, N, W
for surfaceFold in gtList:
    if gtSurfaces is None:
        gtSurfaces = np.load(surfaceFold)
    else:
        gtSurfaces = np.concatenate((gtSurfaces, np.load(surfaceFold)))

# read prediction
predictSurfaces = None
for volumeName in volumeNameList:
    xmlVolumePath=xmlPath+"/"+volumeName+"_Volume_Sequence_Surfaces_Prediction.xml"
    if predictSurfaces is None:
        predictSurfaces = getSurfacesArray(xmlVolumePath)
    else:
        predictSurfaces = np.concatenate((predictSurfaces, getSurfacesArray(xmlVolumePath)))


# statistics
# extract specific surface
gtSurfaces = gtSurfaces[:,sIndex,:]
predictSurfaces = predictSurfaces[:,sIndex,:]
absError = np.abs(predictSurfaces-gtSurfaces)
tableHead = "patientID;allGoodAvg;e<0.25;0.25<=e<0.5;0.5<=e<0.75;0.75<=e<1.0;1.0<=e<1.25;1.25<=e<1.5;1.5<=e<1.75;1.75<=e<2.0; 2.0<=e; \n"

with open(os.path.join(outputPath,outputFile), "w") as file:
    file.write(tableHead)
    for i, patientID in enumerate(patientIDList):
        sliceIndexBox = ['', '', '', '', '', '', '','','']
        lowB = goodBscans[patientID][0] - 1 + i*NumBscansPerPatient;
        highB = goodBscans[patientID][1] + i*NumBscansPerPatient;
        allGoodAvg = f"{np.mean(absError[lowB:highB,]):.2f}"
        for b in range(lowB, highB):
            bMean = np.mean(absError[b,])
            boxIndex = int(bMean//0.25)
            if boxIndex >= len(sliceIndexBox):
                boxIndex = len(sliceIndexBox)-1
            sliceIndexBox[boxIndex] = sliceIndexBox[boxIndex] +str(b-i*NumBscansPerPatient +1)+','
        strIndexBox = ""
        for cell in sliceIndexBox:
            strIndexBox = strIndexBox + cell+";"
        file.write(str(patientID)+";"+allGoodAvg+";"+strIndexBox+"\n")
print("End\n")


