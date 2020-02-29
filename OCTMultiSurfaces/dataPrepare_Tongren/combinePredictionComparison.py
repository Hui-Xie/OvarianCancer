# prediction:
patientID = "440_OD_5194"
OCTSlice = 16
predDir= "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/log/OCTUnetTongren/expUnetTongren_9Surfaces_20200215_CV5/testResult"
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
predImage = imread(predImagePath)
rawImage = predImage[:,0:W]
S = Num_Surfaces


# draw output image
f = plt.figure(frameon=False)
DPI = f.dpi
f.set_size_inches(W*4/float(DPI), H/float(DPI))
subplotRow = 1
subplotCol = 2

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(subplotRow, subplotCol, 1)
subplot1.imshow(predImage, cmap='brg')
subplot1.axis('off')

subplot2 = plt.subplot(subplotRow, subplotCol, 2)
subplot2.imshow(rawImage, cmap='gray')
for s in range(0, S):
    subplot2.plot(range(0, W), patientSurfacesArray[OCTSlice-1, s, :], linewidth=0.4)
subplot2.axis('off')

plt.savefig(os.path.join(outputDir, patientID_Index + "_Image_GT_Predict_OCTExploerer.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()
