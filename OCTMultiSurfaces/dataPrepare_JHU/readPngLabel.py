pngFilePath = "/home/hxie1/data/OCT_JHU/preprocessedData/image/ms14_spectralis_macula_v1_s1_R_16.png"
labelFilePath = "/home/hxie1/data/OCT_JHU/preprocessedData/label/ms14_spectralis_macula_v1_s1_R_16.txt"
outputDir = "/home/hxie1/data/OCT_JHU/output"

import json
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import os



with open(labelFilePath) as json_file:
    surfaces = json.load(json_file)['bds']
surfaces = np.asarray(surfaces)
S,W = surfaces.shape

image = imread(pngFilePath)

patientIDBsan = os.path.splitext(os.path.basename(pngFilePath))[0]

f = plt.figure(frameon=False)
DPI = f.dpi
H = 128
f.set_size_inches(W/ float(DPI), H*2 / float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(2, 1, 1)
subplot1.imshow(image, cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(2, 1, 2)
subplot2.imshow(image, cmap='gray')
for s in range(0, S):
    subplot2.plot(range(0, W), surfaces[s, :], linewidth=0.9)
subplot2.axis('off')

#plt.margins(0)
#plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.
plt.savefig(os.path.join(outputDir, patientIDBsan + "_Raw_GT.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()


print("ok")
