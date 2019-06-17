# data augmentation for images and labels

import os
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
from DataMgr import DataMgr
import math

suffix = "_CT.nrrd"
inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images"
outputImagesDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images_augmt_29_140_140"
outputLabelsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/labels_augmt_23_127_127"
readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images_augmt_29_140_140/readme.txt"

imageGoalSize = (29, 140, 140)
labelGoalSize = (23,127,127)

originalCwd = os.getcwd()
os.chdir(inputsDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputsDir) if suffix in x]
os.chdir(originalCwd)

imageDataMgr = DataMgr("", "", suffix)
imageDataMgr.setDataSize(0, imageGoalSize[0], imageGoalSize[1], imageGoalSize[2], "imageDataAugmentation")
imageRadius = imageGoalSize[0] // 2

labelDataMgr = DataMgr("", "", suffix)
labelDataMgr.setDataSize(0, labelGoalSize[0], labelGoalSize[1], labelGoalSize[2], "labelDataAugmentation")
labelRadius = labelGoalSize[0] // 2

Notes = "Notes:\n"

for file in filesList:
    patientID = DataMgr.getStemName(file, suffix)

    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)

    label = file.replace("_CT.nrrd", "_Seg.nrrd").replace("images/", "labels/")
    label3d = sitk.GetArrayFromImage(sitk.ReadImage(label))

    label3dBinary = (label3d > 0)
    full0Labels = False
    if np.count_nonzero(label3dBinary) ==0:
        full0Labels = True
        massCenter = label3d.shape//2  # mass center is at image center without all 0 labels.
    else:
        massCenterFloat = ndimage.measurements.center_of_mass(label3dBinary)
        massCenter = []
        for i in range(len(massCenterFloat)):
            massCenter.append(int(massCenterFloat[i]))

    # save image
    roi = imageDataMgr.cropVolumeCopy(image3d, massCenter[0], massCenter[1], massCenter[2], imageRadius)
    np.save(os.path.join(outputImagesDir, patientID + "_cc.npy"), roi)

    # save label
    roi = labelDataMgr.cropVolumeCopy(label3d, massCenter[0], massCenter[1], massCenter[2], labelRadius)
    # erase label 3(lymph node)
    roi3 = roi >= 3
    roi[np.nonzero(roi3)] = 0
    np.save(os.path.join(outputLabelsDir, patientID + "_roi.npy"), roi)

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this directory\n")
    f.write(f"goalSize: {imageGoalSize}\n")
    f.write(f"inputDir = {inputsDir}\n")
    f.write(f"inputImagesDir = {outputImagesDir}\n")
    f.write(f"inputLabelsDir = {outputLabelsDir}\n")
    f.write(Notes)