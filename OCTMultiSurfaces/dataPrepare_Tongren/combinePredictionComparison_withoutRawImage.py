# prediction:
#patientID = "440_OD_5194"
#OCTSlice = 16

# 120030_OD_3477_OCT15_Image_GT_Predict.png
patientID = "120030_OD_3477"
OCTSlice = 15

predDir= "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/log/SurfacesUnet/expUnetTongren_20200313_9Surfaces_CV5_Sigma20/testResult"
patientID_Index = patientID+f"_OCT{OCTSlice:02d}"
predImagePath = predDir+ "/"+ patientID_Index+"_Image_GT_Predict.png"
# "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/log/OCTUnetTongren/expUnetTongren_9Surfaces_20200215_CV5/testResult/440_OD_5194_OCT16_Image_GT_Predict.png"

patientSegFile = "/home/hxie1/data/OCT_Tongren/OCTExplorerOutput_W512/Control/"+ patientID+"_Volume_Sequence_Surfaces_Iowa.xml"
outputDir = "/home/hxie1/data/OCT_Tongren/paper"

import sys
sys.path.append(".")
from TongrenFileUtilities import *
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import os


slicesPerPatient = 31
W = 512
H = 496

# read OCTExplorer output
patientSurfacesArray = getSurfacesArray(patientSegFile)

Z, Num_Surfaces, X = patientSurfacesArray.shape
assert 11 == Num_Surfaces and Z == slicesPerPatient
if 11 == Num_Surfaces:
    patientSurfacesArray = np.delete(patientSurfacesArray, 8, axis=1)  # delete inaccurate surface 8
    B, Num_Surfaces, X = patientSurfacesArray.shape
    assert B == 31 and Num_Surfaces == 10

    patientSurfacesArray = np.delete(patientSurfacesArray, 2, axis=1)  # delete inaccurate surface 2
    B, Num_Surfaces, X = patientSurfacesArray.shape
    assert B == 31 and Num_Surfaces == 9

# remove the leftmost and rightmost 128 columns for each B-scans as the segmentation is not accurate
if 768 == X:
    if "5363_OD_25453" == patientID:
        patientSurfacesArray = patientSurfacesArray[:, :, 103:615]  # left shift 25 pixels for case 5363_OD_25453
    else:
        patientSurfacesArray = patientSurfacesArray[:, :, 128:640]

# read prediction image
raWGTPredImage = imread(predImagePath)
rawImage = raWGTPredImage[:,0:W]
gtImage = raWGTPredImage[:,W:2*W]
predImage = raWGTPredImage[:,2*W:]
S = Num_Surfaces


# draw output image
f = plt.figure(frameon=False)
DPI = f.dpi
subplotRow = 1
subplotCol = 3
f.set_size_inches(W*subplotCol/float(DPI), H*subplotRow/float(DPI))


pltColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',  'tab:olive', 'tab:brown', 'tab:pink',  'tab:purple','tab:cyan']
assert S <= len(pltColors)

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

subplot2 = plt.subplot(subplotRow, subplotCol, 1)
subplot2.imshow(gtImage)
subplot2.axis('off')

subplot3 = plt.subplot(subplotRow, subplotCol, 2)
subplot3.imshow(rawImage, cmap='gray')
for s in range(0, S):
    subplot3.plot(range(0, W), patientSurfacesArray[OCTSlice-1, s, :], pltColors[s], linewidth=0.9)
subplot3.axis('off')

subplot4 = plt.subplot(subplotRow, subplotCol, 3)
subplot4.imshow(predImage)
subplot4.axis('off')


plt.savefig(os.path.join(outputDir, patientID_Index + "_GT_OCTExplorer_Predict.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()
