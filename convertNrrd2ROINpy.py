
import os
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
from DataMgr import DataMgr

suffix = "_CT.nrrd"
inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/testImages"
outputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/testImages_ROI_147_281_281"
readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/testImages_ROI_147_281_281/readme.txt"

# inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages"
# outputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_ROI_147_281_281"
# readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_ROI_147_281_281/readme.txt"

goalSize = (147,281,281)

originalCwd = os.getcwd()
os.chdir(inputsDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputsDir) if suffix in x]
os.chdir(originalCwd)

dataMgr = DataMgr("", "", suffix)
dataMgr.setDataSize(0, goalSize[0], goalSize[1],goalSize[2], 0, "ConvertNrrd2ROI")
radius = goalSize[0]//2

for file in filesList:
    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)

    label = DataMgr.getLabelFile(file)
    label3d = sitk.GetArrayFromImage(sitk.ReadImage(label))

    label3d = (label3d > 0)
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


