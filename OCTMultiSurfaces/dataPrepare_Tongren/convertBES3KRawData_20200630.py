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

#todo 
volumesDir = "/home/hxie1/data/OCT_Tongren/Glaucoma"
outputDir = "/home/hxie1/data/OCT_Tongren/numpy/glaucomaRaw_W512"
patientsListFile = os.path.join(outputDir, "patientsList.txt")

def saveVolumeToNumpy(volumesList, goalImageFile, goalPatientsIDFile):
    # image in slices, Heigh, Width axis order
    if len(volumesList) ==0:
        return

    allPatientsImageArray = np.empty((len(volumesList)*NumSlices,H, W), dtype=np.float)
    patientIDDict = {}

    s = 0 # initial slice for each patient
    for volume in volumesList:
        # read image data and clip
        imagesList = glob.glob(volume + f"/*[0-3][0-9].jpg")
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
        random.seed(201910)
        random.shuffle(patientsList)
        saveInputFilesList(patientsList, patientsListFile)

    # do not divide into sublist

    saveVolumeToNumpy(patientsList, os.path.join(outputDir, 'test', f"images.npy"), \
                                           os.path.join(outputDir, 'test', f"patientID.json") )


    print(f"total: {len(patientsList)} patients in {volumesDir}")
    print("===End of prorgram=========")

if __name__ == "__main__":
    main()