
# only output raw image and his ground truth contour.
import os
import sys
import numpy as np
sys.path.append("../..")
from utilities.FilesUtilities import *
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_dilation

imageDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/nrrd_npy"
labelDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/labels_npy"
outputDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/ImageLabelOnly"

suffix = ".npy"
originalCwd = os.getcwd()
os.chdir(imageDir)
filesList = [os.path.abspath(x) for x in os.listdir(imageDir) if suffix in x]
os.chdir(originalCwd)

nFigs = 0
dilateFilter = np.ones((3,3), dtype=int)  # dilation filter for for 4-connected boundary in 2D

if not os.path.exists(outputDir):
    os.makedirs(outputDir)  # recursive dir creation

for file in filesList:
    patientID = getStemName(file, suffix)
    rawFilename = file
    gtFilename = os.path.join(labelDir, patientID + suffix)

    rawImage = np.load(rawFilename).astype(np.float32)
    S,H,W = rawImage.shape
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

    if 0 == numNonzeroSlices:
        print(f"Warn: {patientID} has full zero labels")
        continue

    for s in sliceIndices:
        nFigs += 1
        f = plt.figure(frameon=False)
        DPI = f.dpi
        f.set_size_inches(W / float(DPI), H/ float(DPI))
        plt.margins(0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

        R = rawImage[s,]  # raw image
        G = gtImage[s,].astype(int)  # groundtruth image
        GC = binary_dilation(G != 1, dilateFilter) & G  # groundtruth contour
        GCIndices = np.nonzero(GC)

        '''
        np.set_printoptions(precision=1, threshold=np.inf)
        with open(os.path.join(outputDir, f"output_GC.txt"), "w") as file:
            file.write(np.array2string(GC))
            exit(0)
        '''


        plt.imshow(R, cmap='gray')
        plt.scatter(GCIndices[1], GCIndices[0],s=0.0005)
        plt.axis('off')

        plt.savefig(os.path.join(outputDir, patientID + f"_s{s}.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close()

print(f"totally generated {nFigs} png file. ")