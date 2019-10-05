
import os
import sys
import numpy as np
sys.path.append("..")
from FilesUtilities import *
import matplotlib.pyplot as plt

imageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/nrrd_npy"
labelDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/labels_npy"

suffix = ".npy"
originalCwd = os.getcwd()
os.chdir(imageDir)
filesList = [os.path.abspath(x) for x in os.listdir(imageDir) if suffix in x]
os.chdir(originalCwd)

sliceIndices =[12,25, 37]
nFigs = 0

for file in filesList:
    patientID = getStemName(file, suffix)
    rawFilename = file
    gtFilename = os.path.join(labelDir, patientID + suffix)

    rawImage = np.load(rawFilename).astype(np.float32)
    gtImage = np.load(gtFilename).astype(np.float32)

    for s in sliceIndices:
        nFigs += 1
        f = plt.figure(nFigs)
        subplot1 = plt.subplot(1, 2, 1)
        I = rawImage[s,]
        subplot1.imshow(I, cmap='gray', vmin=np.amin(I), vmax=np.amax(I))
        subplot1.set_title("Raw: " + patientID + f"_s{s}")

        subplot2 = plt.subplot(1, 2, 2)
        I = gtImage[s,] + rawImage[s,]
        subplot2.imshow(I, cmap='gray', vmin=np.amin(I), vmax=np.amax(I))
        subplot2.set_title("GT")

        plt.tight_layout()

        plt.savefig(os.path.join(imageDir, patientID + f"_s{s}.png"))
        plt.close()

print(f"totally generated {len(filesList)*len(sliceIndices)} png file. ")