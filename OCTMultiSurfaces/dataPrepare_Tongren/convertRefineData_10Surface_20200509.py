# convert data and refine data
# according to Dr Yaxing Wang at Tongren direction: current surface 2 has too many manual segmentation error,so we do not consider surface 2 at this time
# at May 9th, 2020:
#       according to PanZhe's slides on April 23th, 2020. GCL/IPL surface is very important to diagnosis glaucoma etc.
#       therefore, we need to restore the 3th surface(ID=2) with BScan range 10-25.
#       remove some patient samples: 1791，4765, and 34127，34169和2579


import glob as glob
import os
import sys
sys.path.append(".")
from TongrenFileUtilities import *
import random
import numpy as np
from imageio import imread
import json
import torch


K = 10   # K-fold Cross validation, the k-fold is for test, (k+1)%K is validation, others for training.
W = 512  # original images have width of 768, we only clip middle 512
H = 496
N = 10  # in xml files, there are 11 surfaces, we will delete inaccurate surface 8
NumSlices = 16  # for each patient, from BScan 10 to Bscan 25, below code index 9 to 25
RemovedPatientIDList=['1791','4765','34127','34169','2579']

volumesDir = "/home/hxie1/data/OCT_Tongren/control"
segsDir = "/home/hxie1/data/OCT_Tongren/refinedGT_20200204"

outputDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet_10Surfaces"
patientsListFile = os.path.join(outputDir, "patientsList.txt")
device = torch.device("cuda:0")

def saveVolumeSurfaceToNumpy(volumesList, goalImageFile, goalSurfaceFile, goalPatientsIDFile):
    # image in slices, Heigh, Width axis order
    # label in slices, NumSurfaces, Width axis order
    if len(volumesList) ==0:
        return

    allPatientsImageArray = np.empty((len(volumesList)*NumSlices,H, W), dtype=np.float)
    allPatientsSurfaceArray = np.empty((len(volumesList)*NumSlices, N, W),dtype=np.int)
    patientIDDict = {}

    s = 0 # initial slice for each patient
    for volume in volumesList:
        patientName = os.path.basename(volume)
        segFile = patientName+"_Sequence_Surfaces_Iowa.xml"
        segFile = os.path.join(segsDir, segFile)

        surfacesArray = getSurfacesArray(segFile)
        Z,Num_Surfaces, X = surfacesArray.shape
        if 11 == Num_Surfaces:
            surfacesArray = np.delete(surfacesArray, 8, axis=1)  # delete inaccurate surface 8
            B, Num_Surfaces, X = surfacesArray.shape
            assert B == 31 and Num_Surfaces == 10 and X == 768

        # remove the leftmost and rightmost 128 columns for each B-scans as the segmentation is not accurate
        if "5363_OD_25453" == patientName:
            surfacesArray = surfacesArray[9:25, :, 103:615]  # left shift 25 pixels for case 5363_OD_25453
        else:
            surfacesArray = surfacesArray[9:25, :, 128:640]
        allPatientsSurfaceArray[s:s+16,:,:] = surfacesArray

        # read image data and clip
        imagesList = glob.glob(volume + f"/*_OCT[0-3][0-9].jpg")
        imagesList.sort()
        if Z != len(imagesList):
           print(f"Error: at {volume}, the slice number does not match jpg files.")
           return

        for z in range(9, 25):
            if "5363_OD_25453" == patientName:
                allPatientsImageArray[s,] = imread(imagesList[z])[:,103:615]
            else:
                allPatientsImageArray[s,] = imread(imagesList[z])[:,128:640]
            patientIDDict[str(s)] = imagesList[z]
            s +=1

    # flip axis order to fit with Leixin's network with format(slices, Width, Height)
    # allPatientsImageArray = np.swapaxes(allPatientsImageArray, 1,2)
    # allPatientsSurfaceArray = np.swapaxes(allPatientsSurfaceArray, 1,2)

    # adjust inaccurate manual segmentation error near foveas
    # use average method along column to correct disorders in surface 0-3 near foveas as excluding surface 2
    surfaces = torch.from_numpy(allPatientsSurfaceArray).to(device)
    surface0 = surfaces[:, :-1, :]
    surface1 = surfaces[:, 1:, :]
    surfacesCopy = torch.zeros_like(surfaces)  # surfaceCopy contain the first disorder values only, it and its next row value along column violate constraints
    surfacesCopy[:, 0:-1, :] = torch.where(surface0 > surface1, surface0, torch.zeros_like(surface0))

    B = surfaces.shape[0]
    MaxMergeSurfaces = 5
    for b in range(0, B):
        for w in range(W * 2 // 5, W * 3 // 5):  # only focus on fovea region.
            s = 0
            while (s < MaxMergeSurfaces):  # in fovea region, the first 4 surfaces (excluding surface2) may merge into one surface.
                if 0 != surfacesCopy[b, s, w]:
                    n = 1  # n continuous disorder locations
                    while s + n + 1 < MaxMergeSurfaces and 0 != surfacesCopy[b, s + n, w]:
                        n += 1
                    # get the average of disorder surfaces
                    correctValue = 0.0
                    for k in range(0, n + 1):  # n+1 consider its next disorder value
                        correctValue += surfaces[b, s + k, w]
                    correctValue /= (n + 1)

                    # fill the average value
                    for k in range(0, n + 1):
                        surfacesCopy[b, s + k, w] = correctValue
                    s = s + n + 1
                else:
                    s += 1
    # after average the disorder near fovea, fill other values which may also be disorder
    surfacesCopy = torch.where(0 == surfacesCopy, surfaces, surfacesCopy)
    # final fully sort along column
    correctedSurfacesArray, _ = torch.sort(surfacesCopy, dim=-2)  # correct all surface constraint
    allPatientsSurfaceArray = correctedSurfacesArray.cpu().numpy()

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
        patientSegsList = glob.glob(segsDir + f"/*_Volume_Sequence_Surfaces_Iowa.xml")
        # from segsList to patientsList
        patientsList = []
        for segName in patientSegsList:
            patientSurfaceName = os.path.splitext(os.path.basename(segName))[0]  # e.g. 1062_OD_9512_Volume_Sequence_Surfaces_Iowa
            patientVolumeName = patientSurfaceName[0:patientSurfaceName.find("_Sequence_Surfaces_Iowa")]  # 1062_OD_9512_Volume
            patientID = patientVolumeName[0:patientVolumeName.find("_OD_")]
            if patientID not in RemovedPatientIDList:
                patientsList.append(volumesDir + f"/{patientVolumeName}")

        # patientsList = glob.glob(volumesDir + f"/*_Volume")  # this is from volume directory start
        print(f"Total {len(patientsList)} patients remaining as deleting 5 ill patients.")
        patientsList.sort()
        random.seed(201910)
        random.shuffle(patientsList)
        saveInputFilesList(patientsList, patientsListFile)

    # split files in sublist, this is a better method than before.
    N = len(patientsList)
    patientsSubList= []
    step = N//K
    for i in range(0,K*step, step):
        nexti = i + step
        patientsSubList.append(patientsList[i:nexti])
    for i in range(K*step, N):
        patientsSubList[i-K*step].append(patientsList[i])


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