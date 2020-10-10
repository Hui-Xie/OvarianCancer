
# check Numpy Slice

slicePath = "/home/hxie1/data/BES_3K/W512AllSlices/499_OS_6108_Slice13.npy"
outputDir = "/home/hxie1/data/temp"



import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(".")

slice = np.load(slicePath)
H,W = slice.shape
print(f"slice.shape = {slice.shape}")

f = plt.figure(frameon=False)
DPI = f.dpi
rowSubplot= 1
colSubplot= 1
f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(slice, cmap='gray')
subplot1.axis('off')


outputFilename, ext = os.path.splitext(os.path.basename(slicePath))
outputFilename = outputFilename + ".png"

plt.savefig(os.path.join(outputDir, outputFilename), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

