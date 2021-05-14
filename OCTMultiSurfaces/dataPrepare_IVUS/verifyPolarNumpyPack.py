

dataSetDir = "/home/hxie1/data/IVUS/polarNumpy_10percent/training"
outputDir = "/home/hxie1/data/temp"

import numpy as np
import os
import json
import matplotlib.pyplot as plt

images = np.load(os.path.join(dataSetDir,"images.npy"))
surfaces = np.load(os.path.join(dataSetDir,"surfaces.npy"))
NSlice1,H,W1 = images.shape
NSlice2,S,W2 =  surfaces.shape
assert NSlice1 ==NSlice2
assert W1==W2
W = W1

print(f"images shape: {images.shape}")
print(f"surfaces shape: {surfaces.shape}")

with open(os.path.join(dataSetDir,"patientID.json")) as json_file:
    patientIDs = json.load(json_file)

k= 6 # random test number

patientID = os.path.splitext(os.path.basename(patientIDs[str(k)]))[0]

f = plt.figure(frameon=False)
# DPI = f.dpi
DPI = 100
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
    subplot2.plot(range(0, W), surfaces[k, s, :], linewidth=0.9)
subplot2.axis('off')

plt.savefig(os.path.join(outputDir, patientID + "_Raw_GT_PolarNumpyVerify.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

