
import os
import SimpleITK as sitk
from FilesUtilities import *
import numpy as np

suffix = "_CT.nrrd"
inputsDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223/nrrd"
outputImagesDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223/numpy"
# outputLabelsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/labels_npy"
readmeFile = "/home/hxie1/data/OvarianCancerCT/pixelSize223/numpy/readme.txt"
# the final assemble size of numpy array
Z,Y,X = 231,251,251

originalCwd = os.getcwd()
os.chdir(inputsDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputsDir) if suffix in x]
os.chdir(originalCwd)

Notes = r"""
        Notes: 
        1  nrrd image is clipped into [0,300] in original intensity;
        2  image normalize into gaussian distribution slice by slice with (x-mu)/std;
        3  image is assembled into fixed size[231,251,251] 
         """

for file in filesList:
    patientID = getStemName(file, suffix)

    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)

    # window level image into [0,300]
    image3d = np.clip(image3d, 0, 300)
    image3d = image3d.astype(float)   # this is very important, otherwise, normalization will be meaningless.

    # normalize image with std  for each slice
    shape = image3d.shape
    for i in range(shape[0]):
        slice = image3d[i,]
        mean = np.mean(slice)
        std  = np.std(slice)
        if 0 != std:
            slice = (slice -mean)/std
        else:
            slice = slice -mean
        image3d[i,] = slice

    # normalize into [0,1]
    # ptp = np.ptp(image3d)
    # image3d = image3d/ptp

    #label = file.replace("_CT.nrrd", "_Seg.nrrd").replace("images/", "labels/")
    #label3d = sitk.GetArrayFromImage(sitk.ReadImage(label))

    # assemble in fixed size[231,251,251] in Z,Y, X direction
    z,y,x = image3d.shape
    wall = np.zeros((Z,Y,X), dtype=np.float)
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
