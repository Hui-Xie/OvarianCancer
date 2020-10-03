# convert BES 3K data around its center 512 A-Scans into Numpy
# and one numpy file contains 31 slices

import glob as glob
import os
import sys
sys.path.append(".")
import numpy as np
from imageio import imread

W = 512  # original images have width of 768, we only clip middle 512
H = 496
NumSlices = 31  # for each patient

volumesDir = "/home/hxie1/data/BES_3K/raw"
outputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes"

def saveVolumeToNumpy():
    patientsList = glob.glob(volumesDir + f"/*_Volume")
    patientsList.sort()

    # check each volume has same number of images
    errorVolumesList = []
    errorVolumesNum = []
    correctVolumesList = []
    for volume in patientsList:
        imagesList = glob.glob(volume + f"/*[0-9][0-9].jpg")
        if NumSlices != len(imagesList):
            errorVolumesList.append(volume)
            errorVolumesNum.append(len(imagesList))
        else:
            correctVolumesList.append(volume)

    with open(os.path.join(outputDir, f"ErrorVolumeList.txt"), "w") as file:
        file.write("ErrorVolumeName, NumBScans,\n")
        num = len(errorVolumesList)
        for i in range(num):
            file.write(f"{os.path.basename(errorVolumesList[i])}, {errorVolumesNum[i]},\n")

    # image in slices, Height, Width axis order
    for volume in correctVolumesList:
        # read image data and clip
        imagesList = glob.glob(volume + f"/*[0-9][0-9].jpg")
        imagesList.sort()
        if NumSlices != len(imagesList):
           print(f"Error: at {volume}, the slice number does not match jpg files.")
           return

        imagesArray = np.empty((NumSlices,H, W), dtype=np.float)
        for s in range(0, NumSlices):
            imagesArray[s,] = imread(imagesList[s])[:,128:640]

        # save
        numpyPath = os.path.join(outputDir, os.path.basename(volume) + ".npy")
        np.save(numpyPath, imagesArray)


def main():
    saveVolumeToNumpy()


if __name__ == "__main__":
    main()
    print("===End of prorgram=========")