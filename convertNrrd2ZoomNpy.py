
import os
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
from DataMgr import DataMgr

suffix = "_CT.nrrd"
#inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/testImages"
#outputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/testImages_zoom_147_281_281"
#readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/testImages_zoom_147_281_281/readme.txt"

inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages"
outputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_zoom_147_281_281"
readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_zoom_147_281_281/readme.txt"

goalSize = (147,281,281)

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
    f.write(f"total {N} files in this directory\n")
    f.write(f"goalSize: {goalSize}\n")
    f.write(f"inputsDir = {inputsDir}\n")


