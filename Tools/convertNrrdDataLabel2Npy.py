
# At same time, convert Nrrd data and label to numpy

inputDataDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223/nrrd"
inputLabelDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223/nrrdLabel"

outputDataDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223withLabel/numpy"
outputLabelDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223withLabel/numpyLabel"
readmeFile = "/home/hxie1/data/OvarianCancerCT/pixelSize223withLabel/readme.txt"

fileIDListFile = "/home/hxie1/data/OvarianCancerCT/pixelSize223withLabel/stdLabelFileList.json"
import json
with open(fileIDListFile) as f:
    fileIDList = json.load(f)

# the final assemble size of numpy array
Z,Y,X = 231,251,251

import sys
import SimpleITK as sitk
sys.path.append("..")
from FilesUtilities import *
import numpy as np

Notes = r"""
        Notes: 
        1  nrrd image is clipped into [0,300] in original intensity;
        2  image normalize into gaussian distribution slice by slice with x/std, a gausssian distributrion with non-zero mean
        3  image is assembled into fixed size[231,251,251] 
        4  sychronize image and label's assemble together
         """

for patientID in fileIDList:
    imageFile = os.path.join(inputDataDir, patientID+"_CT.nrrd")
    labelFile = os.path.join(inputLabelDir, patientID+"_Seg.nrrd")

    image = sitk.ReadImage(imageFile)
    image3d = sitk.GetArrayFromImage(image)
    # window level image into [0,300]
    image3d = np.clip(image3d, 0, 300)
    image3d = image3d.astype(np.float32)   # this is very important, otherwise, normalization will be meaningless.

    label = sitk.ReadImage(labelFile)
    label3d = sitk.GetArrayFromImage(label)

    if image3d.shap != label3d.shape:
        print(f"imageFile: {imageFile} \n labelFile: {labelFile}\n \t have different shape.")
        exit(1)

    # normalize image with std  for each slice
    shape = image3d.shape
    for i in range(shape[0]):
        slice = image3d[i,]
        mean = np.mean(slice)
        std  = np.std(slice)
        if 0 != std:
            # slice = (slice -mean)/std  # gaussian distribution with zero mean
            slice = slice / std  # gaussian distribution with non-zero mean, which will make following padding zero not conflict with the meaning of zero.
        else:
            slice = slice -mean  # if all pixels in a slice equal, they are no discriminating meaning.
        image3d[i,] = slice

    # normalize into [0,1]
    # ptp = np.ptp(image3d)
    # image3d = image3d/ptp

    #label = file.replace("_CT.nrrd", "_Seg.nrrd").replace("images/", "labels/")
    #label3d = sitk.GetArrayFromImage(sitk.ReadImage(label))

    # assemble in fixed size[231,251,251] in Z,Y, X direction
    z,y,x = image3d.shape
    wall = np.zeros((Z,Y,X), dtype=np.float32)
    if z<Z:
        Z1 = (Z-z)//2
        Z2 = Z1+z
        z1 = 0
        z2 = z1+z
    else:
        Z1 = 0
        Z2 = Z1+Z
        z1 = (z-Z)//2
        z2 = z1+Z

    if y < Y:
        Y1 = (Y - y) // 2
        Y2 = Y1 + y
        y1 = 0
        y2 = y1 + y
    else:
        Y1 = 0
        Y2 = Y1 + Y
        y1 = (y - Y) // 2
        y2 = y1 + Y

    if x < X:
        X1 = (X - x) // 2
        X2 = X1 + x
        x1 = 0
        x2 = x1 + x
    else:
        X1 = 0
        X2 = X1 + X
        x1 = (x - X) // 2
        x2 = x1 + X

    wall[Z1:Z2, Y1:Y2,X1:X2] = image3d[z1:z2, y1:y2, x1:x2]


    np.save(os.path.join(outputImagesDir, patientID + ".npy"), wall)
    #np.save(os.path.join(outputLabelsDir, patientID + ".npy"), label3d)

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this directory\n")
    f.write(f"inputDir = {inputsDir}\n")
    f.write(f"inputImagesDir = {outputImagesDir}\n")
    # f.write(f"inputLabelsDir = {outputLabelsDir}\n")
    f.write(Notes)

print("===End of convertNrrd to Npy=======")
