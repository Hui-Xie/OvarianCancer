# convert BES 3K data around its center 512 A-Scans into Numpy
# each patient has one file

import glob as glob
import os
import sys
sys.path.append(".")
from TongrenFileUtilities import *
import random
import numpy as np
from imageio import imread
import json

W = 512  # original images have width of 768, we only clip middle 512
H = 496
NumSlices = 31  # for each patient


volumesDir = "/home/hxie1/data/BES_3K/raw"
outputDir = "/home/hxie1/data/BES_3K/numpy_10SurfaceSeg/W512"
patientsListFile = os.path.join(outputDir, "patientsList.txt")

def saveOneVolumeToNumpy(volumePath, goalImageFile):
    # image in slices, Height, Width axis order
    imageArray = np.empty((NumSlices, H, W), dtype=np.float)
    # read image data and clip
    imagesList = glob.glob(volumePath + f"/*[0-9][0-9].jpg")
    imagesList.sort()
    if NumSlices != len(imagesList):
       print(f"Error: at {volumePath}, the slice number does not match jpg files.")
       return

    for z in range(0, NumSlices):
        imageArray[z,] = imread(imagesList[z])[:,128:640]

    # save
    np.save(goalImageFile, imageArray)

def main():
    # get files list
    if os.path.isfile(patientsListFile):
        patientsList = loadInputFilesList(patientsListFile)
    else:
        patientsList = glob.glob(volumesDir + f"/*_Volume")
        patientsList.sort()

        # check each volume has same number of images
        errorVolumesList= []
        errorVolumesNum = []
        correctVolumesList = []
        for volume in patientsList:
            imagesList = glob.glob(volume + f"/*[0-9][0-9].jpg")
            if NumSlices != len(imagesList):
                errorVolumesList.append(volume)
                errorVolumesNum.append(len(imagesList))
            else:
                correctVolumesList.append(volume)

        with open(os.path.join(outputDir,f"ErrorVolumeList.txt"), "w") as file:
            file.write("ErrorVolumeName, NumBScans,\n")
            num = len(errorVolumesList)
            for i in range(num):
                file.write(f"{os.path.basename(errorVolumesList[i])}, {errorVolumesNum[i]},\n")

        patientsList = correctVolumesList
        saveInputFilesList(patientsList, patientsListFile)

    # split files in sublist, this is a better method than before.
    N = len(patientsList)
    for i in range(N):
        basename = os.path.basename(patientsList[i])
        saveOneVolumeToNumpy(patientsList[i], os.path.join(outputDir, 'testVolume', f"{basename}.npy"))

    print(f"total: {len(patientsList)} patients in {volumesDir}")
    print("===End of prorgram=========")

if __name__ == "__main__":
    main()