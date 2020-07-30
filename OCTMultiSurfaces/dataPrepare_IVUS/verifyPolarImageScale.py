

dataSetDir = "/home/hxie1/data/IVUS/polarNumpy/test"
outputDir = "/home/hxie1/data/temp"

import numpy as np
import os
import json
import matplotlib.pyplot as plt

import sys
sys.path.append("../network")
from OCTAugmentation import *
import torch


images = np.load(os.path.join(dataSetDir,"images.npy"))
surfaces = np.load(os.path.join(dataSetDir,"surfaces.npy"))
NSlice1,H,W1 = images.shape
NSlice2,S,W2 =  surfaces.shape
assert NSlice1 ==NSlice2
assert W1==W2
W = W1

with open(os.path.join(dataSetDir,"patientID.json")) as json_file:
    patientIDs = json.load(json_file)

k= 100 # random test number

patientID = os.path.splitext(os.path.basename(patientIDs[str(k)]))[0]

f = plt.figure(frameon=False)
DPI = f.dpi
rowSubplot= 3
colSubplot= 1
f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(images[k], cmap='gray')
for s in range(0, S):
    subplot1.plot(range(0, W), surfaces[k,s, :], linewidth=0.9)
subplot1.axis('off')

# scale
device = torch.device('cuda:0')
scaleNumerator = 2
scaleDenominator =3
polarImage = torch.from_numpy(images[k]).to(device)
polarLabel = torch.from_numpy(surfaces[k]).to(device)
scaledPolarImage = scalePolarImage(polarImage, scaleNumerator, scaleDenominator)
scaledPolarLabel = scalePolarLabel(polarLabel, scaleNumerator, scaleDenominator)

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(scaledPolarImage.cpu().numpy(), cmap='gray')
for s in range(0, S):
    subplot2.plot(range(0, W), scaledPolarLabel[s, :].cpu().numpy(), linewidth=0.9)
subplot2.axis('off')

# scale back
scaledPolarImage2 = scalePolarImage(scaledPolarImage, scaleDenominator, scaleNumerator)
scaledPolarLabel2 = scalePolarLabel(scaledPolarLabel,scaleDenominator, scaleNumerator)

subplot3 = plt.subplot(rowSubplot,colSubplot, 3)
subplot3.imshow(scaledPolarImage2.cpu().numpy(), cmap='gray')
for s in range(0, S):
    subplot3.plot(range(0, W), scaledPolarLabel2[s, :].cpu().numpy(), linewidth=0.9)
subplot3.axis('off')

plt.savefig(os.path.join(outputDir, patientID + "_Polar_Scale_ScaleBack.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

