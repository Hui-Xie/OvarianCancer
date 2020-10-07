# convert BES 3K data around its center 512 A-Scans into Numpy
# and one numpy file contains the middle only

import glob as glob
import os
import sys
sys.path.append(".")
import numpy as np
from imageio import imread

W = 512  # original images have width of 768, we only clip middle 512
H = 496

volumesDir = "/home/hxie1/data/BES_3K/raw"
outputDir = "/home/hxie1/data/BES_3K/W512MidSlices"

def saveSliceToNumpy():
    patientsList = glob.glob(volumesDir + f"/*_Volume")
    patientsList.sort()

    # image in slices, Height, Width axis order
    for volume in patientsList:
        # read image data and clip
        imagesList = glob.glob(volume + f"/*[0-9][0-9].jpg")
        imagesList.sort()
        numSlices != len(imagesList)

        mid = numSlices//2
        imageArray = imread(imagesList[mid])[:,128:640]

        # save
        b = os.path.basename(volume)
        outputFilename = b[0:b.rfind("Volume")] +f"midSlice.npy"
        numpyPath = os.path.join(outputDir, outputFilename)
        np.save(numpyPath, imageArray)


def main():
    saveSliceToNumpy()


if __name__ == "__main__":
    main()
    print("===End of prorgram=========")