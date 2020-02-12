# read IVUS image and label

imagePath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/DCM/frame_02_0001_003.png"
lumenLabelPath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS/lum_frame_02_0001_003.txt"
mediaLabelPath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS/med_frame_02_0001_003.txt"
outputDir = "/home/hxie1/temp"


import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
import os
from numpy import genfromtxt

image = imread(imagePath).astype(np.float32)
H,W = image.shape

lumenLabel = genfromtxt(lumenLabelPath, delimiter=',')
mediaLabel = genfromtxt(mediaLabelPath, delimiter=',')

patientID = os.path.splitext(os.path.basename(imagePath))[0]

f = plt.figure(frameon=False)
DPI = f.dpi
f.set_size_inches(W*3/ float(DPI), H/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(1, 3, 1)
subplot1.imshow(image, cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(1, 3, 2)
subplot2.imshow(image, cmap='gray')
subplot2.plot(lumenLabel[:,0], lumenLabel[:,1], linewidth=0.4)
subplot2.plot(mediaLabel[:,0], mediaLabel[:,1], linewidth=0.4)
subplot2.axis('off')

# verify the grond truth order
subplot3 = plt.subplot(1, 3, 3)
subplot3.imshow(image, cmap='gray')
for i in range(0, 360, 45):
    subplot3.text(lumenLabel[i,0], lumenLabel[i,1], str(i//45), color = 'blue')
    subplot3.text(mediaLabel[i,0], mediaLabel[i,1], str(i//45), color = 'red')
subplot3.axis('off')


plt.savefig(os.path.join(outputDir, patientID + "_Raw_GT.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

print("========Program Ends=============")

