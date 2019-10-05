

# convert Nrrd images and labels to numpy array without zoom.
# keep same physical size

import sys
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
sys.path.append("..")
from FilesUtilities import *

import matplotlib.pyplot as plt


suffix = ".nrrd"
inputImageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/nrrd"
inputLabelDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/labels"
outputImageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/nrrd_npy"
outputLabelDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/labels_npy"
readmeFile = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/nrrd_npy/readme.txt"

originalCwd = os.getcwd()
os.chdir(inputImageDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputImageDir) if suffix in x]
os.chdir(originalCwd)

flipAxis = (1,2)

for file in filesList:
    patientID = getStemName(file, ".nrrd")
    # for image data
    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)
    image3d = np.clip(image3d, -100, 250)  # window level
    image3d = image3d.astype(np.float32)  # this is very important, otherwise, normalization will be meaningless.
    imageShape = image3d.shape

    # normalize image for whole volume
    mean = np.mean(image3d)
    std  = np.std(image3d)
    image3d = (image3d-mean)/std

    image3d = np.flip(image3d,flipAxis)  # keep numpy image has same RAS direction with Nrrd image.

    np.save(os.path.join(outputImageDir, patientID + ".npy"), image3d)

    # for label
    labelFile = os.path.join(inputLabelDir, patientID + "-label.nrrd")
    label = sitk.ReadImage(labelFile)
    label3d = sitk.GetArrayFromImage(label)
    label3d = label3d.astype(np.float32)  # this is very important
    labelShape = label3d.shape
    if labelShape != imageShape:
        print(f"Error: images shape != label shape for {file} and {labelFile} ")
        exit(1)
    label3d = np.flip(label3d, flipAxis)

    np.save(os.path.join(outputLabelDir, patientID + ".npy"), label3d)

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this directory\n")
    f.write(f"inputsDir = {inputImageDir}\n")
    f.write(f"all images keeps its original size \n")
    f.write("All numpy image filp along (1,2) axis to keep RAS orientation consistent with Nrrd.\n")

print(f"totally convert {N} files")



