# convert BES 3K data around its center 512 A-Scans into Numpy

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
K = 200 # total 6565 volumes, we divide them into 200 packages.


volumesDir = "/home/hxie1/data/BES_3K/raw"
outputDir = "/home/hxie1/data/BES_3K/numpy/W512"
patientsListFile = os.path.join(outputDir, "patientsList.txt")

def saveVolumeToNumpy(volumesList, goalImageFile, goalPatientsIDFile):
    # image in slices, Height, Width axis order
    if len(volumesList) ==0:
        return

    allPatientsImageArray = np.empty((len(volumesList)*NumSlices,H, W), dtype=float)
    patientIDDict = {}

    s = 0 # initial slice for each patient
    for volume in volumesList:
        # read image data and clip
        imagesList = glob.glob(volume + f"/*[0-9][0-9].jpg")
        imagesList.sort()
        if NumSlices != len(imagesList):
           print(f"Error: at {volume}, the slice number does not match jpg files.")
           return

        for z in range(0, NumSlices):
            allPatientsImageArray[s,] = imread(imagesList[z])[:,128:640]
            patientIDDict[str(s)] = imagesList[z]
            s +=1

    # save
    np.save(goalImageFile, allPatientsImageArray)
    with open(goalPatientsIDFile, 'w') as fp:
        json.dump(patientIDDict, fp)

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
    patientsSubList = []
    step = N // K
    for i in range(0, K * step, step):
        nexti = i + step
        patientsSubList.append(patientsList[i:nexti])
    for i in range(K * step, N):
        patientsSubList[i - K * step].append(patientsList[i])

    for i in range(K):
        saveVolumeToNumpy(patientsSubList[i], os.path.join(outputDir, 'testPackage', f"images_{i}.npy"), \
                                               os.path.join(outputDir, 'testPackage', f"patientID_{i}.json") )

    print(f"total: {len(patientsList)} patients in {volumesDir}")
    print("===End of prorgram=========")

if __name__ == "__main__":
    main()