
import os
import sys
import numpy as np
sys.path.append("..")
from FilesUtilities import *
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation

imageDir = "/home/hxie1/data/OvarianCancerCT/primaryROISmall/nrrd_npy"
labelDir = "/home/hxie1/data/OvarianCancerCT/primaryROISmall/labels_npy"

suffix = ".npy"
originalCwd = os.getcwd()
os.chdir(imageDir)
filesList = [os.path.abspath(x) for x in os.listdir(imageDir) if suffix in x]
os.chdir(originalCwd)

nFigs = 0
dilateFilter = np.ones((3,3), dtype=int)  # dilation filter for for 4-connected boundary in 2D

for file in filesList:
    patientID = getStemName(file, suffix)
    rawFilename = file
    gtFilename = os.path.join(labelDir, patientID + suffix)

    rawImage = np.load(rawFilename).astype(np.float32)
    gtImage = np.load(gtFilename).astype(np.float32)

    # get sliceIndices for gtImage.
    nonzeroIndex = np.nonzero(gtImage)
    nonzeroSlices = list(map(int, set(nonzeroIndex[0])))  # make sure the slice index is int.
    numNonzeroSlices = len(nonzeroSlices)
    sliceIndices = []
    if numNonzeroSlices <= 5:
        sliceIndices = nonzeroSlices
    else:
        step = numNonzeroSlices // 6
        for i in range(step, numNonzeroSlices, step):
            sliceIndices.append(nonzeroSlices[i])

    for s in sliceIndices:
        nFigs += 1
        f = plt.figure(nFigs)

        R = rawImage[s,]  # raw image
        G = gtImage[s,].astype(int)  # groundtruth image
        GC = binary_dilation(G != 1, dilateFilter) & G  # groundtruth contour

        subplot1 = plt.subplot(1, 2, 1)
        subplot1.imshow(R, cmap='gray', vmin=np.amin(R), vmax=np.amax(R))
        subplot1.set_title("Raw: " + patientID + f"_s{s}")

        subplot2 = plt.subplot(1, 2, 2)
        subplot2.imshow(R, vmin=np.amin(R), vmax=np.amax(R))
        subplot2.imshow(GC, cmap='YlGn', alpha=0.3, vmin=np.amin(GC), vmax=np.amax(GC))
        subplot2.set_title("GroundTruth Contour")

        plt.tight_layout()

        plt.savefig(os.path.join(imageDir, patientID + f"_s{s}.png"))
        plt.close()

print(f"totally generated {nFigs} png file. ")