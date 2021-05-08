

imagesFile = "/home/hxie1/data/OCT_Duke/numpy_10Percent/training/images.npy"
surfacesFile = imagesFile.replace("/images.", "/surfaces.")
patientIDFile = "/home/hxie1/data/OCT_Duke/numpy_10Percent/training/patientID.json"
outputDir = "/home/hxie1/data/temp"


import numpy as np
import os
import matplotlib.pyplot as plt
import json

k = 170


images = np.load(imagesFile)
surfaces = np.load(surfacesFile)
N, H,W1 = images.shape
N, S,W2 =  surfaces.shape
assert W1==W2
assert S ==3
W = W1
print(f"images.shaep = {images.shape}")
print(f"surfaces.shape = {surfaces.shape}")

with open(patientIDFile) as f:
    IDs = json.load(f)


patientID_Index= os.path.splitext(IDs[str(k)])[0]

f = plt.figure(frameon=False)
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
subplot2.imshow(images[k], cmap='gray')
surface = surfaces[k]
for s in range(0, S):
    subplot2.plot(range(0, W), surface[s, :], linewidth=0.9)
subplot2.axis('off')

outputPath = os.path.join(outputDir,patientID_Index + "_Raw_GT.png")
plt.savefig(outputPath, dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

print(f"output image: {outputPath}")

