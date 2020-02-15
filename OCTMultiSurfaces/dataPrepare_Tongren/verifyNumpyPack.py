

dataSetDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/validation"
outputDir = "/home/hxie1/temp"

kf=5 # k fold

import numpy as np
import os
import json
import matplotlib.pyplot as plt

import sys
sys.path.append(".")
from TongrenFileUtilities import extractFileName

images = np.load(os.path.join(dataSetDir,f"images_CV{kf:d}.npy"))
surfaces = np.load(os.path.join(dataSetDir,f"surfaces_CV{kf:d}.npy"))
NSlice1,H,W1 = images.shape
NSlice2,S,W2 =  surfaces.shape
assert NSlice1 ==NSlice2
assert W1==W2
assert S ==9 # as delete inaccurate surface 8 and 2.
W = W1

with open(os.path.join(dataSetDir,f"patientID_CV{kf:d}.json")) as json_file:
    patientIDs = json.load(json_file)

k= 16 # random test number

#example: "/home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
patientID_Index = extractFileName(patientIDs[str(k)])  # e.g.: 4511_OD_29134_OCT06

f = plt.figure(frameon=False)
DPI = f.dpi
rowSubplot= 1
colSubplot= 2
f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(images[k], cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(images[k,], cmap='gray')
for s in range(0, S):
    subplot2.plot(range(0, W), surfaces[k, s, :], linewidth=0.4)
subplot2.axis('off')

plt.savefig(os.path.join(outputDir, patientID_Index + "_Raw_GT.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

