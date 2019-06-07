
import os
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
from DataMgr import DataMgr
import math

suffix = "_CT.nrrd"
inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images"
outputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/Images_ROI_29_140_140"
readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/Images_ROI_29_140_140/readme.txt"

goalSize = (29,140,140)

originalCwd = os.getcwd()
os.chdir(inputsDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputsDir) if suffix in x]
os.chdir(originalCwd)

dataMgr = DataMgr("", "", suffix)
dataMgr.setDataSize(0, goalSize[0], goalSize[1],goalSize[2], "ConvertNrrd2ROI")
radius = goalSize[0]//2

Notes = "Exception files without available labels: \n"

for file in filesList:
    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)

    label = file.replace("_CT.nrrd", "_Seg.nrrd").replace("images/", "labels/")
    label3d = sitk.GetArrayFromImage(sitk.ReadImage(label))

    label3d = (label3d > 0)
    if np.count_nonzero(label3d) ==0:
        Notes += f"\t {file}\n"
        continue

    massCenterFloat = ndimage.measurements.center_of_mass(label3d)
    massCenter = []
    for i in range(len(massCenterFloat)):
        massCenter.append(int(massCenterFloat[i]))

    roi = dataMgr.cropVolumeCopy(image3d, massCenter[0], massCenter[1],  massCenter[2], radius)
    patientID = DataMgr.getStemName(file, suffix)
    np.save(os.path.join(outputsDir, patientID + "_roi.npy"), roi)

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this directory\n")
    f.write(f"goalSize: {goalSize}\n")
    f.write(f"inputsDir = {inputsDir}\n")
    f.write(Notes)


