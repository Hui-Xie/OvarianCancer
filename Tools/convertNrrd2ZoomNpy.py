
# convert Nrrd images and labels to numpy array with zoom.

import sys
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
sys.path.append("..")
from FilesUtilities import *

import matplotlib.pyplot as plt


suffix = "_pri.nrrd"
inputImageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI/nrrd"
inputLabelDir = "/home/hxie1/data/OvarianCancerCT/primaryROI/labels"
outputImageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI/nrrd_npy"
outputLabelDir = "/home/hxie1/data/OvarianCancerCT/primaryROI/labels_npy"
readmeFile = "/home/hxie1/data/OvarianCancerCT/primaryROI/nrrd_npy/readme.txt"

goalSize = (51,171,171) # Z,Y,X in nrrd axis order

originalCwd = os.getcwd()
os.chdir(inputImageDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputImageDir) if suffix in x]
os.chdir(originalCwd)

flipAxis = (1,2)

for file in filesList:
    patientID = getStemName(file, "_pri.nrrd")
    # for image data
    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)
    image3d = np.clip(image3d, -100, 250)  # window level
    image3d = image3d.astype(np.float32)  # this is very important, otherwise, normalization will be meaningless.
    imageShape = image3d.shape
    zoomFactor = [goalSize[0] / imageShape[0], goalSize[1] / imageShape[1], goalSize[2] / imageShape[2]]
    image3d = ndimage.zoom(image3d, zoomFactor)

    # normalize image for whole volume
    mean = np.mean(image3d)
    std  = np.std(image3d)
    image3d = (image3d-mean)/std

    image3d = np.flip(image3d,flipAxis)  # keep numpy image has same RAS direction with Nrrd image.

    np.save(os.path.join(outputImageDir, patientID + ".npy"), image3d)

    # for label
    labelFile = os.path.join(inputLabelDir, patientID + "_pri_seg.nrrd")
    label = sitk.ReadImage(labelFile)
    label3d = sitk.GetArrayFromImage(label)
    label3d = label3d.astype(np.float32)  # this is very important
    labelShape = label3d.shape
    if labelShape != imageShape:
        print(f"Error: images shape != label shape for {file} and {labelFile} ")
        exit(1)

    label3d = ndimage.zoom(label3d, zoomFactor)
    label3d = (label3d > 0.1).astype(np.float32)
    label3d = np.flip(label3d, flipAxis)

    np.save(os.path.join(outputLabelDir, patientID + ".npy"), label3d)

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this directory\n")
    f.write(f"inputsDir = {inputImageDir}\n")
    f.write(f"all images are zoom into a same size: {goalSize}\n")
    f.write("label image first zoom by spline and then judge whether value >0.1 to get new label.\n")
    f.write("All numpy image filp along (1,2) axis to keep RAS orientation consistent with Nrrd.\n")

print(f"totally convert {N} files")



