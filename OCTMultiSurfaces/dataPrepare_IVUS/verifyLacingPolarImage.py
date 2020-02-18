# verify lace and delace polar image and label

dataSetDir = "/home/hxie1/data/IVUS/polarNumpy/test"
outputDir = "/home/hxie1/data/temp"

import numpy as np
import os
import json
import matplotlib.pyplot as plt

import torch
import sys
sys.path.append("../network")
from OCTAugmentation import *

images = np.load(os.path.join(dataSetDir,"images.npy"))
surfaces = np.load(os.path.join(dataSetDir,"surfaces.npy"))
NSlice1,H,W1 = images.shape
NSlice2,S,W2 =  surfaces.shape
assert NSlice1 ==NSlice2
assert W1==W2
W = W1

with open(os.path.join(dataSetDir,"patientID.json")) as json_file:
    patientIDs = json.load(json_file)

k= 53 # random test number
lacingWidth = 350

patientID = os.path.splitext(os.path.basename(patientIDs[str(k)]))[0]

f = plt.figure(frameon=False)
DPI = f.dpi
rowSubplot= 3
colSubplot= 1
lacedW = W+2*lacingWidth
f.set_size_inches(lacedW*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(images[k], cmap='gray')
for s in range(0, S):
    subplot1.plot(range(0, W), surfaces[k, s, :], linewidth=0.4)
subplot1.axis('off')

imagek = torch.from_numpy(images[k])
surfacek = torch.from_numpy(surfaces[k])

lacedImagek, lacedSurfacek = lacePolarImageLabel(imagek, surfacek, lacingWidth)


subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(lacedImagek.numpy(), cmap='gray')
for s in range(0, S):
    subplot2.plot(range(0, lacedW), lacedSurfacek[s, :].numpy(), linewidth=0.4)
subplot2.axis('off')

delacedImagek, delacedSurfacek = delacePolarImageLabel(lacedImagek, lacedSurfacek, lacingWidth)

subplot3 = plt.subplot(rowSubplot,colSubplot, 3)
subplot3.imshow(delacedImagek.numpy(), cmap='gray')
for s in range(0, S):
    subplot3.plot(range(0, W), delacedSurfacek[s, :].numpy(), linewidth=0.4)
subplot3.axis('off')


plt.savefig(os.path.join(outputDir, patientID + "_Raw_lace350_delace.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

