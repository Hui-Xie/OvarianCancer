# convert BES 3K data around its center 512 A-Scans into Numpy
# convert each slice into a numpy file

import glob as glob
import os
import sys
sys.path.append(".")
import numpy as np
from imageio import imread

W = 512  # original images have width of 768, we only clip middle 512
H = 496

volumesDir = "/home/hxie1/data/BES_3K/raw"
outputDir  = "/home/hxie1/data/BES_3K/W512AllSlices"

def saveAllSlicesToNumpy():
    patientsList = glob.glob(volumesDir + f"/*_Volume")
    patientsList.sort()

    # check each volume has same number of images
    volumesList = []
    volumeNumSlices = []
    for volume in patientsList:
        imagesList = glob.glob(volume + f"/*[0-9][0-9].jpg")
        volumesList.append(volume)
        volumeNumSlices.append(len(imagesList))

    with open(os.path.join(outputDir, f"volume_NumSlices.csv"), "w") as file:
        file.write("volumeName,NumBScans,\n")
        num = len(volumesList)
        for i in range(num):
            file.write(f"{os.path.basename(volumesList[i])},{volumeNumSlices[i]},\n")

    # image in slices, Height, Width axis order
    for volume in volumesList:
        # read image data and clip
        imagesList = glob.glob(volume + f"/*[0-9][0-9].jpg")
        imagesList.sort()
        numSlices = len(imagesList)

        for i in range(numSlices):
            imageArray = imread(imagesList[i])[:, 128:640]
            # save
            b = os.path.basename(volume)
            outputFilename = b[0:b.rfind("Volume")] + f"Slice{i:02d}.npy"
            numpyPath = os.path.join(outputDir, outputFilename)
            np.save(numpyPath, imageArray)

def main():
    saveAllSlicesToNumpy()


if __name__ == "__main__":
    main()
    print("===End of prorgram of convert all slices into numpy files. =========")