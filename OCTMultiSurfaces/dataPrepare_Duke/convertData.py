# Extract Duke data as input for deep learning network

# for each volume:
# patient:
#        images: [512×1000×100 double]: Height*Width*Slice
#     layerMaps: [100×1000×3 double]  :
#             Slice*Widht* NumSurface,
# extract middle volume: 51 slices, height 512, Width 361,

# data division, similar with Leixin:
# Control: 18 Test, 18 Validation, 79 training;  Total 115
# AMD:     41 Test, 41 Validation, 187 training; Total 269

# output:
# in test, validation, and training subdirectories;
# each patient has 2 files: images_ID.npy and surfaces_ID.npy;
# a total index file in each subdirectory;


import glob as glob
import os
import random
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import center_of_mass

AMDPath = "/home/hxie1/data/OCT_Duke/AMD"
ControlPath = "/home/hxie1/data/OCT_Duke/Control"
outputPath = "/home/hxie1/data/OCT_Duke/numpy"

# the size of center volume.
S,H,W= 51,512,361
N = 3

def saveVolumeSurfaceToNumpy(volumesList, outputDir):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)  # recursive dir creation

    epsilon = 1e-8

    patientList=[]
    discardedList= []
    extrapolateList = []
    for volumeFile in volumesList:
        # read mat, and convert it into numpy
        patient = loadmat(volumeFile)
        rawImages = patient['images'] # size:512x1000x100
        rawImages = np.transpose(rawImages,[2,0,1]) # size: 100x512x1000, BxHxW
        rawSurfaces = patient['layerMaps'] #size: 100x1000x3
        rawSurfaces = np.transpose(rawSurfaces, [0,2,1]) # size: 100x3x1000, BxNxW

        # get center of all labeled surface
        C_tuple = center_of_mass((rawSurfaces==rawSurfaces).astype(np.int))
        C = []
        for x in C_tuple:
            C.append(int(x))

        # crop around labelCenter: S,H,W= 51,512,361
        halfS = S//2
        halfW = W//2
        surfaces = rawSurfaces[C[0]-halfS:C[0]+halfS+1, :, C[2]-halfW:C[2]+halfW+1] #size: Sx3xW
        images = rawImages[C[0]-halfS:C[0]+halfS+1, :, C[2]-halfW:C[2]+halfW+1]  # size: Sx512xW
        assert N == surfaces.shape[1]


        # fixed crop
        # extract middle volume: 60x512x400: 60 slices, height 512, Width 400,
        # slice index: 20-80, height index:0-512, width index: 300-700, index start from 0
        # images = rawImages[20:80, :, 300:700]  # size: 60x512x400
        # S,H,W = images.shape
        #  surfaces = rawSurfaces[20:80,:, 300:700] # size: 60x3x400
        #  N = surfaces.shape[1]

        # check nan
        if np.isnan(np.sum(images)):
            print(f"image {volumeFile} has a center volume with nan values. discarded it")
            discardedList.append(volumeFile)
            continue

        # extrapolate nan
        discardThisVolume = False
        extrapolate  = False
        if np.isnan(np.sum(surfaces)):
            for s in range(S):
                if discardThisVolume:
                    break
                for n in range(N):
                    if not np.all(surfaces[s,n,:] == surfaces[s,n,:]):
                        non_nanLocation = np.argwhere(surfaces[s,n,:] == surfaces[s,n,:])
                        non_nanLocation = np.squeeze(non_nanLocation, axis=1)
                        if np.size(non_nanLocation)< 0.8*W:
                            print(f"{volumeFile} has more than 0.2*W nan labels at s={s}, n={n} surface. discarded it")
                            discardedList.append(volumeFile)
                            discardThisVolume = True
                            break
                        low = non_nanLocation[0]
                        high = non_nanLocation[-1]
                        extrapolate = True
                        for w in range(W):
                            if np.isnan(surfaces[s,n,w]) and w < low:
                                surfaces[s, n, w] = surfaces[s, n, low]
                            if np.isnan(surfaces[s, n, w]) and w > high:
                                surfaces[s, n, w] = surfaces[s, n, high]
        if discardThisVolume:
            continue

        if np.isnan(np.sum(surfaces)):
            print(f"After exterploate of nan, {volumeFile} still nan labels. discarded it")
            discardedList.append(volumeFile)
            continue

        if extrapolate:
            extrapolateList.append(volumeFile)

        assert  not np.isnan(np.sum(images)) and not np.isnan(np.sum(surfaces))

        # normalize image
        mean = np.mean(images,(1,2),keepdims=True)
        std  = np.std(images, (1,2),keepdims=True)
        mean = np.repeat(mean, [H],axis=1)
        std = np.repeat(std, [H], axis=1)
        mean = np.repeat(mean, [W], axis=2)
        std = np.repeat(std, [W], axis=2)
        images = (images - mean)/(std+epsilon)

        # construct file name
        basename = os.path.basename(volumeFile) # Farsiu_Ophthalmology_2013_Control_Subject_1001.mat
        basename = os.path.splitext(basename)[0]
        patientID = basename[basename.rfind('_')+1:]
        p = basename[0:basename.rfind('_Subject_')]  # Farsiu_Ophthalmology_2013_Control
        disease = p[p.rfind('_')+1:]
        filename = disease+'_'+patientID
        outputFile = os.path.join(outputDir,filename)

        # save to dir
        np.save(outputFile+"_images.npy", images)
        np.save(outputFile+"_surfaces.npy", surfaces)

        # add fileList
        patientList.append(outputFile+"_images.npy")

    # output information
    with open(os.path.join(outputDir,"patientList.txt"),'w') as f:
        for file in patientList:
            f.write(file + "\n")
    with open(os.path.join(outputDir,"discardedList.txt"),'w') as f:
        for file in discardedList:
            f.write(file + "\n")
    with open(os.path.join(outputDir,"extrapolateList.txt"),'w') as f:
        for file in extrapolateList:
            f.write(file + "\n")

    print(f"Converted {len(patientList)} volumes, extrapolated {len(extrapolateList)} volumes, and discarded {len(discardedList)} volumes, in {outputDir}.")


def main():
    # get files list
    AMDList = glob.glob(AMDPath + f"/*.mat")
    ControlList = glob.glob(ControlPath+f"/*.mat")

    AMDList.sort()
    ControlList.sort()
    random.seed(202008)
    random.shuffle(AMDList)
    random.shuffle(ControlList)

    # divide training, validation and test set
    testList= AMDList[0:41]+ControlList[0:18]
    validationList = AMDList[41:82]+ControlList[18:36]
    trainingList = AMDList[82:]+ControlList[36:]

    saveVolumeSurfaceToNumpy(testList,outputPath+"/test")
    saveVolumeSurfaceToNumpy(validationList, outputPath + "/validation")
    saveVolumeSurfaceToNumpy(trainingList, outputPath + "/training")

    print("====== End of Duke data convert to Numpy ===========")


if __name__ == "__main__":
    main()