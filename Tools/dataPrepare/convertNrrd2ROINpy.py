

# convert Nrrd images and labels to numpy array
# keep same physical size for file of size (49,147,147) without zoom
# if file size is different with (49,147,147), scale it to (49,147,147)
# if ROI includes label2, repress it into 0.



import sys
import SimpleITK as sitk
from scipy import ndimage
import numpy as np
sys.path.append("..")
from utilities.FilesUtilities import *

suffix = ".nrrd"
inputImageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/nrrd"
inputLabelDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/labels"
outputImageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/nrrd_npy"
outputLabelDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/labels_npy"
readmeFile = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/nrrd_npy/readme.txt"

goalSize = (49,147,147) # Z,Y,X in nrrd axis order for primaryROI dir

originalCwd = os.getcwd()
os.chdir(inputImageDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputImageDir) if suffix in x]
os.chdir(originalCwd)

flipAxis = (1,2)

zoomedFileList = []

for file in filesList:
    patientID = getStemName(file, ".nrrd")
    # read image data
    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)

    image3d = np.clip(image3d, -135, 215)  # standard window level 350/40 in 3D slicer.

    image3d = image3d.astype(np.float32)  # this is very important, otherwise, normalization will be meaningless.
    imageShape = image3d.shape

    if imageShape != goalSize:
        zoomFactor = [goalSize[0] / imageShape[0], goalSize[1] / imageShape[1], goalSize[2] / imageShape[2]]
        image3d = ndimage.zoom(image3d, zoomFactor, order=3)
        print(f"ID: {patientID} zoomed to {goalSize}")
        zoomedFileList.append(patientID)

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
    # repress label 2, only keep label 1. As some primary cancer is neighboring with label 2 metastases which is also in ROI.
    label3d = (label3d ==1).astype(np.float32) # this is very important
    labelShape = label3d.shape
    if labelShape != imageShape:
        print(f"Error: images shape != label shape for {file} and {labelFile} ")
        exit(1)
    if labelShape != goalSize:
        label3d = ndimage.zoom(label3d, zoomFactor, order=0)  # nearest neighbor interpolation

    label3d = np.flip(label3d, flipAxis)



    np.save(os.path.join(outputLabelDir, patientID + ".npy"), label3d)

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this directory\n")
    f.write(f"inputsDir = {inputImageDir}\n")
    f.write(f"all images keeps its original size of {goalSize}, except: files {zoomedFileList}\n")
    f.write("All numpy image filp along (1,2) axis to keep RAS orientation consistent with Nrrd.\n")

print(f"totally convert {N} files")



