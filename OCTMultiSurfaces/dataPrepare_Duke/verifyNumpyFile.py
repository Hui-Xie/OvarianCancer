

imagesFile = "/home/hxie1/data/OCT_Duke/numpy/test/AMD_1031_images.npy"
surfacesFile = imagesFile.replace("_images.npy", "_surfaces.npy")
outputDir = "/home/hxie1/data/temp"


import numpy as np
import os
import matplotlib.pyplot as plt



images = np.load(imagesFile)
surfaces = np.load(surfacesFile)
NSlice1,H,W1 = images.shape
NSlice2,S,W2 =  surfaces.shape
assert NSlice1 ==NSlice2
assert W1==W2
assert S ==3
W = W1

k= 45 # random test number

patientID_Index= imagesFile[0:imagesFile.rfind('.npy')]+f"_S{k}"
patientID_Index = os.path.basename(patientID_Index)

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
    subplot2.plot(range(0, W), surfaces[k, s, :], linewidth=0.9)
subplot2.axis('off')

plt.savefig(os.path.join(outputDir,patientID_Index + "_Raw_GT.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

