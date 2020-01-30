
import glob as glob
import os
import sys
sys.path.append(".")
from FileUtilities import *
import random
import numpy as np
from imageio import imread
import json


K = 10  # K-fold Cross validation, the k-fold is for test, (k+1)%K is validation, others for training.

W = 768
H = 496
NumSurfaces = 11
NumSlices = 31  # for each patient

volumesDir = "/home/hxie1/data/OCT_Tongren/control"
segsDir = "/home/hxie1/data/OCT_Tongren/Correcting_Seg"

outputDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet"
patientsListFile = os.path.join(outputDir, "patientsList.txt")

def saveVolumeSurfaceToNumpy(volumesList, goalImageFile, goalSurfaceFile, goalPatientsIDFile):
    # image in slices, Heigh, Width axis order
    # label in slices, NumSurfaces, Width axis order
    if len(volumesList) ==0:
        return

    allPatientsImageArray = np.empty((len(volumesList)*NumSlices,H, W), dtype=np.float)
    allPatientsSurfaceArray = np.empty((len(volumesList)*NumSlices, NumSurfaces, W),dtype=np.int)
    patientIDDict = {}

    s = 0 # initial slice for each patient
    for volume in volumesList:
        segFile = os.path.basename(volume)+"_Sequence_Surfaces_Iowa.xml"
        segFile = os.path.join(segsDir, segFile)

        surfacesArray = getSurfacesArray(segFile)
        Z,surfaces_num, X = surfacesArray.shape
        assert X == W and surfaces_num == NumSurfaces
        allPatientsSurfaceArray[s:s+Z,:,:] = surfacesArray

        # read image data
        imagesList = glob.glob(volume + f"/*_OCT[0-3][0-9].jpg")
        imagesList.sort()
        if Z != len(imagesList):
           print(f"Error: at {volumesList}, the slice number does not match jpg files.")
           return

        for z in range(0, Z):
            allPatientsImageArray[s,] = imread(imagesList[z])
            patientIDDict[str(s)] = imagesList[z]
            s +=1

    # remove the leftmost and rightmost 128 columns for each B-scans as the segmentation is not accurate
    allPatientsImageArray = allPatientsImageArray[:,:,128:640]
    allPatientsSurfaceArray = allPatientsSurfaceArray[:,:,128:640]

    # flip axis order to fit with Leixin's network with format(slices, Width, Height)
    # allPatientsImageArray = np.swapaxes(allPatientsImageArray, 1,2)
    # allPatientsSurfaceArray = np.swapaxes(allPatientsSurfaceArray, 1,2)

    # gaurantee surface separation constraints, s_i <= s_{i+1} in each A-Scan.
    # above error is only 1.6% of all pixel points in Tongren data, direct sort in each A-Scan to gaurantee surface order.
    allPatientsSurfaceArray = np.sort(allPatientsSurfaceArray, axis=-2)

    # save
    np.save(goalImageFile, allPatientsImageArray)
    np.save(goalSurfaceFile, allPatientsSurfaceArray)
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

    # split files in sublist
    N = len(patientsList)
    patientsSubList= []
    step = int(N/K+0.5)
    for i in range(0,N, step):
        nexti = i + step
        if nexti > N:
            nexti = N
        patientsSubList.append(patientsList[i:nexti])
        if nexti == N:
            break

    # partition for test, validation, and training
    outputValidation = True

    for k in range(0,K):
        partitions = {}
        partitions["test"] = patientsSubList[k]

        if outputValidation:
            k1 = (k + 1) % K  # validation k
            partitions["validation"] = patientsSubList[k1]
        else:
            k1 = k
            partitions["validation"] = []

        partitions["training"] = []
        for i in range(K):
            if i != k and i != k1:
                partitions["training"] += patientsSubList[i]

        # save to file
        saveVolumeSurfaceToNumpy(partitions["test"], os.path.join(outputDir, 'test', f"images_CV{k}.npy"),\
                                                     os.path.join(outputDir, 'test', f"surfaces_CV{k}.npy"), \
                                                     os.path.join(outputDir, 'test', f"patientID_CV{k}.json"))
        if outputValidation:
            saveVolumeSurfaceToNumpy(partitions["validation"], os.path.join(outputDir, 'validation', f"images_CV{k}.npy"), \
                                                           os.path.join(outputDir, 'validation', f"surfaces_CV{k}.npy"), \
                                                           os.path.join(outputDir, 'validation', f"patientID_CV{k}.json") )
        saveVolumeSurfaceToNumpy(partitions["training"], os.path.join(outputDir, 'training', f"images_CV{k}.npy"), \
                                                         os.path.join(outputDir, 'training', f"surfaces_CV{k}.npy"), \
                                                         os.path.join(outputDir, 'training', f"patientID_CV{k}.json") )


        print(f"in CV: {k}/{K}: test: {len(partitions['test'])} patients;  validation: {len(partitions['validation'])} patients;  training: {len(partitions['training'])} patients, ")


    print("===End of prorgram=========")

if __name__ == "__main__":
    main()