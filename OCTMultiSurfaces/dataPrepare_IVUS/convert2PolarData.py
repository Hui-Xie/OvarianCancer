#  convert IVUS data into proper numpy pacakage for network usage
#  training set: 109 slices, from which we randomly extract 9 for validation
#  test set: 326 slices;

import glob as glob
import os
import sys
import random
import numpy as np
from imageio import imread
import json

# raw original image size
W = 384
H = 384
NumSurfaces = 2
NumSlices = 49  # the number of B-Scans for each patient

# for training
imagesDir = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/DCM"
segsDir = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS"

# todo: for test

outputDir = "/home/sheen/c-xwu000_data/IVUS/ploarNumpy"

def saveVolumeSurfaceToNumpy(imagesList, goalImageFile, goalSurfaceFile, goalPatientsIDFile):
    # image in (slices, Heigh, Width) axis order
    # label in (slices, C, N, 2) axis order
    if len(imagesList) ==0:
        return

    allPatientsImageArray = np.empty((len(imagesList) , H, W), dtype=np.float)
    allPatientsSurfaceArray = np.empty((len(imagesList), NumSurfaces, W), dtype=np.float) # the ground truth of JHU data is float
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
    imagesList = glob.glob(imagesDir + f"/frame_*_003.png")  # frame_05_0004_003.png
    imagesList.sort()

    N = len(imagesList)
    if "Training" in imagesDir: # from Training data, randomly extract 9 for validation
        assert N==109
        fullList = list(range(N))
        validationIndices = random.sample(fullList, 9)
        validationImageList = []
        for i in validationIndices:
            validationImageList.append(imagesList[i])
        for i in validationIndices:
            del imagesList[i]
        trainingImageList = imagesList

        print(f"train set has {len(trainingImageList)} files, while validation set has {len(validationImageList)} files.")

        saveVolumeSurfaceToNumpy(trainingImageList, os.path.join(outputDir, 'training', f"images.npy"), \
                                 os.path.join(outputDir, 'training', f"surfaces.npy"), \
                                 os.path.join(outputDir, 'training', f"patientID.json"))
        saveVolumeSurfaceToNumpy(validationImageList, os.path.join(outputDir, 'validation', f"images.npy"), \
                                 os.path.join(outputDir, 'validation', f"surfaces.npy"), \
                                 os.path.join(outputDir, 'validation', f"patientID.json"))

    else:
        testImageList = imagesList
        print(f"test set has {len(testImageList)} files.")

        saveVolumeSurfaceToNumpy(testImageList, os.path.join(outputDir, 'test', f"images.npy"), \
                                 os.path.join(outputDir, 'test', f"surfaces.npy"), \
                                 os.path.join(outputDir, 'test', f"patientID.json"))


    print("===End of prorgram=========")

if __name__ == "__main__":
    main()