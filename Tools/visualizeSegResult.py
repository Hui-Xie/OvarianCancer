
# visualize prediction result
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from FilesUtilities import *

def main():
    imageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/nrrd_npy"
    groundTruthDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/labels_npy"
    predictDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/predictResult"
    diceFile = os.path.join(predictDir, "predict_CV5_20191011_170356.txt")   # need to modify

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

        for line in lines:
            line = line.replace('\t\t','\t')
            row = line.split('\t')
            if len(row) ==2:
                ID_Dice[row[0]] = float(row[1])


    nFigs = 0
    for file in filesList:
        patientID = getStemName(file, suffix)
        rawFilename = os.path.join(imageDir, patientID+suffix)
        gtFilename  = os.path.join(groundTruthDir, patientID+suffix)
        predictFilename = file

        rawImage = np.load(rawFilename).astype(np.float32)
        gtImage = np.load(gtFilename).astype(np.float32)
        predictImage = np.load(predictFilename).astype(np.float32)

        # get sliceIndices for gtImage.
        nonzeroIndex = np.nonzero(gtImage)
        nonzeroSlices = list(map(int, set(nonzeroIndex[0])))  # make sure the slice index is int.
        numNonzeroSlices = len(nonzeroSlices)
        sliceIndices = []
        if numNonzeroSlices <= 5:
            sliceIndices = nonzeroSlices
        else:
            step = numNonzeroSlices//6
            for i in range(step, numNonzeroSlices, step):
                sliceIndices.append(nonzeroSlices[i])

        for s in sliceIndices:
            nFigs +=1
            f = plt.figure(nFigs)
            subplot1 = plt.subplot(2,2,1)
            I = rawImage[s,]
            subplot1.imshow(I, cmap='gray', vmin=np.amin(I), vmax=np.amax(I))
            subplot1.set_title("Raw: "+ patientID+f"_s{s}")

            subplot2 = plt.subplot(2,2,2)
            I = rawImage[s,] + gtImage[s,]
            subplot2.imshow(I, cmap='gray', vmin=np.amin(I), vmax=np.amax(I))
            subplot2.set_title("GroundTruth")

            subplot3 = plt.subplot(2,2,3)
            I = rawImage[s,] + predictImage[s,]
            subplot3.imshow(I, cmap='gray', vmin=np.amin(I), vmax=np.amax(I))
            dice = float(ID_Dice[patientID])
            subplot3.set_title(f"Predict_Dice({dice: .2%})")

            subplot4 = plt.subplot(2,2,4)
            I = rawImage[s,]+ (gtImage[s,]-predictImage[s,])
            subplot4.imshow(I, cmap='gray', vmin=np.amin(I), vmax=np.amax(I))
            subplot4.set_title("GT-Predict")

            plt.tight_layout()

            plt.savefig(os.path.join(predictDir, patientID+f"_s{s}.png"))
            plt.close()



    print(f"Totoally save {nFigs} image file in {predictDir}")


if __name__ == "__main__":
    main()
