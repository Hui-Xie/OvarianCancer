
# visualize prediction result
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from FilesUtilities import *

def getDiceFromArray(array, patientID):
    nRows = array.shape[0]
    for row in range(nRows):
        if array[row, 0] == patientID:
            return array[row, 1]

def main():
    imageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI/nrrd_npy"
    groundTruthDir = "/home/hxie1/data/OvarianCancerCT/primaryROI/labels_npy"
    predictDir = "/home/hxie1/data/OvarianCancerCT/primaryROI/predictionResult"
    diceFile = os.path.join(predictDir, "predict_CV0_20190916_163347.txt")

    suffix = ".npy"
    originalCwd = os.getcwd()
    os.chdir(predictDir)
    filesList = [os.path.abspath(x) for x in os.listdir(predictDir) if suffix in x]
    os.chdir(originalCwd)

    # read dice data
    ID_Dice={}
    with open(diceFile) as file:
        data = file.read()
        lines = data.splitlines()
        lines.pop(0)   #erase tabel head.

        countRow = 0
        for line in lines:
            line = line.replace('\t\t','\t')
            row = line.split('\t')
            if len(row) ==2:
                ID_Dice[row[0]] = row[1]


    sliceIndices = [12,18, 25, 31, 37]

    nFigs = 0

    flipAxis = (1,2)

    for file in filesList:
        patientID = getStemName(file, suffix)
        rawFilename = os.path.join(imageDir, patientID+suffix)
        gtFilename  = os.path.join(groundTruthDir, patientID+suffix)
        predictFilename = file

        rawImage = np.flip(np.load(rawFilename).astype(np.float32),flipAxis)
        gtImage = np.flip(np.load(gtFilename).astype(np.float32),flipAxis)
        predictImage = np.flip(np.load(predictFilename).astype(np.float32),flipAxis)


        for s in sliceIndices:
            nFigs +=1
            f = plt.figure(nFigs)
            subplot1 = plt.subplot(2,2,1)
            subplot1.imshow(rawImage[s,], cmap='gray')
            subplot1.set_title("rawImage")

            subplot2 = plt.subplot(2,2,2)
            subplot2.imshow(gtImage[s,]+rawImage[s,] )  # need to set 0 for label pixels in raw image.
            subplot2.set_title("GT")

            subplot3 = plt.subplot(2,2,3)
            subplot3.imshow(predictImage[s,]+rawImage[s,] )
            dice = float(ID_Dice[patientID])
            subplot3.set_title(f"predict_Dice({dice: .2%})")

            subplot4 = plt.subplot(2,2,4)
            subplot4.imshow(gtImage[s,]-  predictImage[s,]+ rawImage[s,])
            subplot4.set_title("GT-predict")

            # f.suptitle(patientID + "_dice_0.98")

            plt.tight_layout()

            plt.savefig(os.path.join(predictDir, patientID+f"_s{s}.png"))
            plt.close()



    print(f"Totoally save {nFigs} image file in {predictDir}")


if __name__ == "__main__":
    main()
