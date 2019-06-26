# list all mass center for each labeled slice, and save them into a dictionary

import os
import DataMgr
import numpy as np

suffix = ".npy"
inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/labels_npy"
massCenterFileName = "massCenterForEachLabeledSlice.json"

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
    for s in nonzeroSlices:

    if np.count_nonzero(label3d) == 0:
        Notes += f"\t {file}\n"
        continue
    massCenterFloat = ndimage.measurements.center_of_mass(label3dB)
    massCenter = []
    for i in range(len(massCenterFloat)):
        massCenter.append(int(massCenterFloat[i]))