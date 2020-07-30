
dataSetDir = "/home/hxie1/data/OCT_JHU/numpy/training"
outputDir = "/home/hxie1/data/OCT_JHU/output"

import numpy as np
import os
import json
import matplotlib.pyplot as plt

images = np.load(os.path.join(dataSetDir,"images.npy"))
surfaces = np.load(os.path.join(dataSetDir,"surfaces.npy"))

with open(os.path.join(dataSetDir,"patientID.json")) as json_file:
    patientIDs = json.load(json_file)

k= 350 # random test number

patientIDBsan = os.path.splitext(os.path.basename(patientIDs[str(k)]))[0]

f = plt.figure(frameon=False)
DPI = f.dpi
H = 128
W = 1024
S = 9
f.set_size_inches(W/ float(DPI), H*2 / float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(2, 1, 1)
subplot1.imshow(images[k], cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(2, 1, 2)
subplot2.imshow(images[k,], cmap='gray')
for s in range(0, S):
    subplot2.plot(range(0, W), surfaces[k, s, :], linewidth=0.9)
subplot2.axis('off')

plt.savefig(os.path.join(outputDir, patientIDBsan + "_Raw_GT_numpyVerify.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()
