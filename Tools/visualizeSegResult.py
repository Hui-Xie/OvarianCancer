
# visualize prediction result
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from FilesUtilities import *
from scipy.ndimage.morphology import binary_dilation
import  json

imageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/nrrd_npy"
groundTruthDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/labels_npy"
predictDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/predict"
outputDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/predict/visualResult"
suffix = ".npy"

def main():
    diceFile = os.path.join(predictDir, "patientDice.json")   # need to modify

    originalCwd = os.getcwd()
    os.chdir(predictDir)
    filesList = [os.path.abspath(x) for x in os.listdir(predictDir) if suffix in x]
    os.chdir(originalCwd)

    if not os.path.exists(outputDir):
        os.mkdir(outputDir)

    # read dice data
    ID_Dice = {}
    with open(diceFile) as f:
        ID_Dice = json.load(f)

    dilateFilter = np.ones((3,3), dtype=int)  # dilation filter for for 4-connected boundary in 2D
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

            R = rawImage[s,]  # raw image
            G = gtImage[s,].astype(int)   # groundtruth image
            P = predictImage[s,].astype(int)  # predicted image
            GC =  binary_dilation(G!=1, dilateFilter) & G  # groundtruth contour
            PC =  binary_dilation(P!=1, dilateFilter) & P  # predicted contour

            subplot1 = plt.subplot(2,2,1)
            subplot1.imshow(R, cmap='gray', vmin=np.amin(R), vmax=np.amax(R))
            subplot1.set_title("Raw: "+ patientID+f"_s{s}")

            subplot2 = plt.subplot(2,2,2)
            subplot2.imshow(R, cmap='gray', vmin=np.amin(R), vmax=np.amax(R))
            subplot2.imshow(GC,cmap='YlGn', alpha= 0.3, vmin=np.amin(GC), vmax=np.amax(GC))
            subplot2.set_title("GroundTruth Contour")

            subplot3 = plt.subplot(2,2,3)
            subplot3.imshow(R, cmap='gray', vmin=np.amin(R), vmax=np.amax(R))
            subplot3.imshow(PC, cmap='YlOrRd', alpha=0.3, vmin=np.amin(PC), vmax=np.amax(PC))
            dice = float(ID_Dice[patientID])
            subplot3.set_title(f"Predict_Dice({dice: .2%})")

            subplot4 = plt.subplot(2,2,4)
            subplot4.imshow(R, cmap='gray', vmin=np.amin(R), vmax=np.amax(R))
            subplot4.imshow(GC, cmap='YlGn', alpha=0.3, vmin=np.amin(GC), vmax=np.amax(GC) )
            subplot4.imshow(PC, cmap='YlOrRd', alpha=0.3, vmin=np.amin(PC), vmax=np.amax(PC))
            subplot4.set_title("GT and Predict")

            plt.tight_layout()

            plt.savefig(os.path.join(outputDir, patientID+f"_s{s}.png"))
            plt.close()




    print(f"Totoally save {nFigs} image file in {predictDir}")


if __name__ == "__main__":
    main()
