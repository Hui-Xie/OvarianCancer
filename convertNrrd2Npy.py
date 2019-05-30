
import os
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
from DataMgr import DataMgr

inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages"
suffix = "_CT.nrrd"
outputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_zoom"
readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_zoom/readme.txt"

goalSize = (73,141,141)

originalCwd = os.getcwd()
os.chdir(inputsDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputsDir) if suffix in x]
os.chdir(originalCwd)

for file in filesList:
    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)
    shape = image3d.shape
    zoomFactor = [goalSize[0] / shape[0], goalSize[1] / shape[1], goalSize[2] / shape[2]]
    image3d = ndimage.zoom(image3d, zoomFactor)
    patientID = DataMgr.getStemName(file, suffix)
    np.save(os.path.join(outputsDir, patientID + "_zoom.npy"), image3d)

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this directory")
    f.write(f"goalSize: {goalSize}")
    f.write(f"inputsDir = {inputsDir}")


