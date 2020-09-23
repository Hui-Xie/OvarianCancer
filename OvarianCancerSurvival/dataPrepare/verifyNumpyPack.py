


numpyFile = "/home/hxie1/data/OvarianCancerCT/rawNrrd/images_H281_W281/96260660_CT.npy"
outputDir = "/home/hxie1/data/temp"


import numpy as np
import os
import matplotlib.pyplot as plt

import sys
sys.path.append(".")

images = np.load(numpyFile)
S,H,W = images.shape
print(f"images.shape = {images.shape}")

sliceA = 50
sliceB = 150

f = plt.figure(frameon=False)
DPI = f.dpi
rowSubplot= 1
colSubplot= 2
f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(images[sliceA,], cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(images[sliceB,], cmap='gray')
subplot2.axis('off')

plt.savefig(os.path.join(outputDir, "verfiyNumpySliceA_SliceB.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

