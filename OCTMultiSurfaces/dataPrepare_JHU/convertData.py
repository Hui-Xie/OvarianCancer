# "We train on the last 6 HC and last 9 MS subjects and test on the other 20 subjects. "
#  convert JHU data into proper numpy pacakage for network usage


import glob as glob
import os
import sys
import random
import numpy as np
from imageio import imread
import json

W = 1024  # the number of A-Scans
H = 128
NumSurfaces = 9
NumSlices = 49  # the number of B-Scans for each patient

imagesDir = "/home/hxie1/data/OCT_JHU/preprocessedData/image"
segsDir = "/home/hxie1/data/OCT_JHU/preprocessedData/correctedLabel"

outputDir = "/home/hxie1/data/OCT_JHU/numpy"

def saveVolumeSurfaceToNumpy(imagesList, goalImageFile, goalSurfaceFile, goalPatientsIDFile):
    # image in slices, Height, Width axis order
    # label in slices, NumSurfaces, Width axis order
    if len(imagesList) ==0:
        return

    allPatientsImageArray = np.empty((len(imagesList) , H, W), dtype=float)
    allPatientsSurfaceArray = np.empty((len(imagesList), NumSurfaces, W), dtype=float) # the ground truth of JHU data is float
    patientIDDict = {}

    s = 0 # initial slice index
    for imagePath in imagesList:
        patientIDBsan = os.path.splitext(os.path.basename(imagePath))[0]
        segFile = os.path.join(segsDir, patientIDBsan+".json")

        with open(segFile) as json_file:
            surfaces = json.load(json_file)['bds']
        surfacesArray = np.asarray(surfaces)

        surfaces_num, X = surfacesArray.shape
        assert X == W and surfaces_num == NumSurfaces
        allPatientsSurfaceArray[s,:,:] = surfacesArray

        # read image data
        # JHU data has been normalize in its Matlab preprocessing
        allPatientsImageArray[s,] = imread(imagePath)
        patientIDDict[str(s)] = imagePath
        s +=1

    # save
    np.save(goalImageFile, allPatientsImageArray)
    np.save(goalSurfaceFile, allPatientsSurfaceArray)
    with open(goalPatientsIDFile, 'w') as fp:
        json.dump(patientIDDict, fp)


def main():
    # get files list
    imagesList = glob.glob(imagesDir + f"/*.png")
    imagesList.sort()

    # split files in sublist: train and test:
    # "We train on the last 6 HC and last 9 MS subjects and test on the other 20 subjects. "
    #  it is consistent experiment config with Fully Convolutional Boundary Regression for Retina OCT Segmentation (MICCAI2019)
    N = len(imagesList)
    print(f"total {N} images files")
    imagesTrainList= []
    imagesValidationList= []
    for imagePath in imagesList:
        patientIDBsan = os.path.splitext(os.path.basename(imagePath))[0]
        patientID = patientIDBsan[0:4] #hc09 or ms04
        if patientID[0:2] == 'hc':
            if int(patientID[2:4]) <= 8:
                imagesValidationList.append(imagePath)
            else:
                imagesTrainList.append(imagePath)
        else: # 'ms'
            if int(patientID[2:4]) <= 12:
                imagesValidationList.append(imagePath)
            else:
                imagesTrainList.append(imagePath)

    print(f"train set has {len(imagesTrainList)} files, while validation set has {len(imagesValidationList)} files")



    # save to file
    saveVolumeSurfaceToNumpy(imagesValidationList, os.path.join(outputDir, 'validation', f"images.npy"),\
                                                 os.path.join(outputDir, 'validation', f"surfaces.npy"), \
                                                 os.path.join(outputDir, 'validation', f"patientID.json"))

    saveVolumeSurfaceToNumpy(imagesTrainList, os.path.join(outputDir, 'training', f"images.npy"), \
                                                     os.path.join(outputDir, 'training', f"surfaces.npy"), \
                                                     os.path.join(outputDir, 'training', f"patientID.json") )

    print("===End of prorgram=========")

if __name__ == "__main__":
    main()