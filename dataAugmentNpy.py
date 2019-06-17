# data augmentation for images and labels

import os
import SimpleITK as sitk
import numpy as np
from DataMgr import DataMgr

suffix = "_CT.nrrd"
inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images"
outputImagesDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images_augmt_29_140_140"
outputLabelsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/labels_augmt_23_127_127"
readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images_augmt_29_140_140/readme.txt"

imageGoalSize = (29, 140, 140)
labelGoalSize = (23, 127, 127)
wRadius = 70
hRadius = 70
imageRadius = imageGoalSize[0] // 2
labelRadius = labelGoalSize[0] // 2

# assume the original images and labels has same size.
originalCwd = os.getcwd()
os.chdir(inputsDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputsDir) if suffix in x]
os.chdir(originalCwd)

imageDataMgr = DataMgr("", "", suffix)
imageDataMgr.setDataSize(0, imageGoalSize[0], imageGoalSize[1], imageGoalSize[2], "imageDataAugmentation")

labelDataMgr = DataMgr("", "", suffix)
labelDataMgr.setDataSize(0, labelGoalSize[0], labelGoalSize[1], labelGoalSize[2], "labelDataAugmentation")

Notes = "Notes:\n"
counter = 0

for file in filesList:
    patientID = DataMgr.getStemName(file, suffix)

    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)

    label = file.replace("_CT.nrrd", "_Seg.nrrd").replace("images/", "labels/")
    label3d = sitk.GetArrayFromImage(sitk.ReadImage(label))

    # label3dBinary = (label3d > 0)
    # full0Labels = False
    # if np.count_nonzero(label3dBinary) ==0:
    #     full0Labels = True
    #     massCenter = label3d.shape//2  # mass center is at image center without all 0 labels.
    # else:
    #     massCenterFloat = ndimage.measurements.center_of_mass(label3dBinary)
    #     massCenter = []
    #     for i in range(len(massCenterFloat)):
    #         massCenter.append(int(massCenterFloat[i]))

    shape = image3d.shape

    for x in range(imageRadius, shape[0]- imageRadius):
        for y in range(hRadius, shape[1]-hRadius):
            for z in range(wRadius, shape[2]-wRadius):
                fileSuffix = f"_sc{x:03d}_{y:03d}_{z:03d}.npy"  # sc means sliding center
                # save image
                roi = imageDataMgr.cropVolumeCopy(image3d, x,y,z, imageRadius)
                np.save(os.path.join(outputImagesDir, patientID + fileSuffix), roi)

                # save label
                roi = labelDataMgr.cropVolumeCopy(label3d,  x,y,z, labelRadius)
                roi3 = roi >= 3
                roi[np.nonzero(roi3)] = 0  # erase label 3(lymph node)
                np.save(os.path.join(outputLabelsDir, patientID + fileSuffix), roi)

                counter +=1

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this input directory\n")
    f.write(f"goalImageSize: {imageGoalSize}\n")
    f.write(f"goalLabelsSize:{labelGoalSize}\n")
    f.write(f"inputDir = {inputsDir}\n")
    f.write(f"inputImagesDir = {outputImagesDir}\n")
    f.write(f"inputLabelsDir = {outputLabelsDir}\n")
    f.write(f"total output {counter} image files and its corresponding label files.\n")
    f.write(Notes)

print(f"total output {counter} image files and its corresponding label files.\n")
