# list all mass center for each labeled slice, and save them into a dictionary

import os
import DataMgr
import numpy as np
from scipy import ndimage
import json


suffix = ".npy"
inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/labels_npy"
massCenterFileName = "massCenterForEachLabeledSlice.json"
outputFilePath = os.path.join(inputsDir, massCenterFileName)

massCenterDict = {}

originalCwd = os.getcwd()
os.chdir(inputsDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputsDir) if suffix in x]
os.chdir(originalCwd)

for file in filesList:
    patientID = DataMgr.getStemName(file, suffix)
    label3d = np.load(file)
    shape = label3d.shape

    label3dB = ((label3d<3) & (label3d >0))  # label3D binary version without considering the nymph node

    nonzeroIndex = np.nonzero(label3dB)
    nonzeroSlices = set(nonzeroIndex[0])
    massCenterList = []
    for s in nonzeroSlices:
        massCenter = ndimage.measurements.center_of_mass(label3dB[s])
        massCenterList.append((s,)+massCenter)

    massCenterDict[patientID] = massCenterList

# output dictionary
jsonData = json.dumps(massCenterDict)
f = open(outputFilePath,"w")
f.write(jsonData)
f.close()

print(f"==============Output: {outputFilePath}========")