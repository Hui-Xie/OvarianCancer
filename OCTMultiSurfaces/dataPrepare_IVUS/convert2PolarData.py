#  convert IVUS data into proper numpy pacakage for network usage
#  training set: 109 slices, from which we randomly extract 9 for validation
#  test set: 326 slices;

import glob as glob
import os
import sys
import random
import numpy as np
from imageio import imread
from numpy import genfromtxt
import json
import sys
sys.path.append(".")
from PolarCartesianConverter import PolarCartesianConverter

# raw original image size
rawW = 384
rawH = 384
C = 2  # number of contour for each image

# polar image size
W = 360 #N
H = rawW//2


imagesDir = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/DCM"

# for training
segsDir = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS"
# todo: for test

outputDir = "/home/hxie1/data/IVUS/ploarNumpy"

def saveVolumeSurfaceToNumpy(imagesList, goalImageFile, goalSurfaceFile, goalPatientsIDFile):
    # polar image in (slices, Heigh, Width) axis order
    # polar label in (slices, C, W) axis order, where N is 360 representing degree from 0 to 359
    if len(imagesList) ==0:
        return

    allPatientsImageArray = np.empty((len(imagesList) , H, W), dtype=np.float)
    allPatientsSurfaceArray = np.empty((len(imagesList), C, W), dtype=np.float) # the ground truth is float
    patientIDDict = {}

    imageShape = (rawH, rawW)
    polarConverter = PolarCartesianConverter(imageShape, rawW // 2, rawH // 2, min(W // 2, H // 2), 360)

    s = 0 # initial slice index
    for imagePath in imagesList:
        patientID = os.path.splitext(os.path.basename(imagePath))[0]  # frame_05_0004_003
        lumenSegFile = os.path.join(segsDir, "lum_"+patientID+".txt")       # lum_frame_01_0001_003.txt
        mediaSegFile = os.path.join(segsDir, "med_"+patientID + ".txt")   # e.g. med_frame_01_0030_003.txt
        lumenLabel = genfromtxt(lumenSegFile, delimiter=',')
        mediaLabel = genfromtxt(mediaSegFile, delimiter=',')

        cartesianLabel = np.array([lumenLabel, mediaLabel])
        cartesianImage = imread(imagePath).astype(np.float32)

        polarImage, polarLabel = polarConverter.cartesianImageLabel2Polar(cartesianImage, cartesianLabel, rotation=0)
        assert (H,W) == polarImage.shape
        assert (C,W) == polarLabel.shape

        allPatientsSurfaceArray[s,:,:] = polarLabel
        allPatientsImageArray[s,] = polarImage
        patientIDDict[str(s)] = imagePath
        s +=1

    # save
    np.save(goalImageFile, allPatientsImageArray)
    np.save(goalSurfaceFile, allPatientsSurfaceArray)
    with open(goalPatientsIDFile, 'w') as fp:
        json.dump(patientIDDict, fp)


def main():
    # get files list
    segsList = glob.glob(segsDir + f"/lum_frame_*_003.txt")  # lum_frame_01_0004_003.txt
    segsList.sort()

    N = len(segsList)
    if "Training" in segsList: # from Training data, randomly extract 9 for validation
        assert N==109

        fullList = list(range(N))
        validationIndices = random.sample(fullList, 9)
        validationImageList = []
        trainingImageList = []

        for i in range(N):
            patientID = os.path.splitext(os.path.basename(segsList[i]))[0]  # lum_frame_05_0004_003
            patientID = patientID[4:]
            imagePath = os.path.join(imagesDir,patientID+".png") # frame_01_0004_003.png

            if i in validationIndices:
                validationImageList.append(imagePath)
            else:
                trainingImageList.append(imagePath)

        print(f"train set has {len(trainingImageList)} files, while validation set has {len(validationImageList)} files.")

        saveVolumeSurfaceToNumpy(trainingImageList, os.path.join(outputDir, 'training', f"images.npy"), \
                                 os.path.join(outputDir, 'training', f"surfaces.npy"), \
                                 os.path.join(outputDir, 'training', f"patientID.json"))
        saveVolumeSurfaceToNumpy(validationImageList, os.path.join(outputDir, 'validation', f"images.npy"), \
                                 os.path.join(outputDir, 'validation', f"surfaces.npy"), \
                                 os.path.join(outputDir, 'validation', f"patientID.json"))

    else:
        assert N==326
        testImageList = []
        for i in range(N):
            patientID = os.path.splitext(os.path.basename(segsList[i]))[0]  # lum_frame_05_0004_003
            patientID = patientID[4:]
            imagePath = os.path.join(imagesDir,patientID+".png") # frame_01_0004_003.png
            testImageList.append(imagePath)


        print(f"test set has {len(testImageList)} files.")

        saveVolumeSurfaceToNumpy(testImageList, os.path.join(outputDir, 'test', f"images.npy"), \
                                 os.path.join(outputDir, 'test', f"surfaces.npy"), \
                                 os.path.join(outputDir, 'test', f"patientID.json"))


    print("===End of prorgram=========")

if __name__ == "__main__":
    main()