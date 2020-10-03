
# check Numpy Volume

volumePath = "/home/hxie1/data/BES_3K/W512NumpyVolumes/1047_OD_8911_Volume.npy"
outputDir = "/home/hxie1/data/temp"



import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(".")

volume = np.load(volumePath)
S, H,W = volume.shape
print(f"volume.shape = {volume.shape}")

# random test slice
s1 = 15
s2 = 25

f = plt.figure(frameon=False)
DPI = f.dpi
rowSubplot= 1
colSubplot= 2
f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(volume[s1], cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(volume[s2], cmap='gray')
subplot2.axis('off')

outputFilename, ext = os.path.split(os.path.basename(volumePath))
outputFilename +="_rand2slices.png"

plt.savefig(os.path.join(outputDir, outputFilename), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

