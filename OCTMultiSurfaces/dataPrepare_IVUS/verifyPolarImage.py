# verify polar image image and label

imagePath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/DCM/frame_01_0001_003.png"
lumenLabelPath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS/lum_frame_01_0001_003.txt"
mediaLabelPath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS/med_frame_01_0001_003.txt"
outputDir = "/home/hxie1/temp"


import sys
sys.path.append(".")
from PolarCoordinate import PolarCoordinate




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

f1 = plt.figure(frameon=False)
DPI = f1.dpi
rowSubplot= 1
colSubplot= 3
f1.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))
plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(image, cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(image, cmap='gray')
subplot2.plot(lumenLabel[:,0], lumenLabel[:,1], linewidth=0.4)
subplot2.plot(mediaLabel[:,0], mediaLabel[:,1], linewidth=0.4)
subplot2.axis('off')

# verify the grond truth order
subplot3 = plt.subplot(rowSubplot,colSubplot, 3)
subplot3.imshow(image, cmap='gray')
for i in range(0, 360, 45):
    subplot3.text(lumenLabel[i,0], lumenLabel[i,1], str(i//45), color = 'blue')
    subplot3.text(mediaLabel[i,0], mediaLabel[i,1], str(i//45), color = 'red')
subplot3.axis('off')

plt.savefig(os.path.join(outputDir, patientID + "_Raw_GT.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()


# convert polar image
polarConverter = PolarCoordinate(W//2,H//2,min(W//2,H//2), 360)
label = np.array([lumenLabel,mediaLabel])

polarImage, polarLabel = polarConverter.cartesianImageLabel2Polar(image,label,rotation=0)

f2 = plt.figure(frameon=False)
DPI = f2.dpi
H,W = polarImage.shape
rowSubplot= 1
colSubplot= 2
f2.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))
plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(polarImage, cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(polarImage, cmap='gray')
subplot2.plot(polarLabel[0,:,0], polarLabel[0,:,1], linewidth=0.4)
subplot2.plot(polarLabel[1,:,0], polarLabel[1,:,1], linewidth=0.4)
subplot2.axis('off')

plt.savefig(os.path.join(outputDir, patientID + "_polar.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()


# convert polar image  with rotation
polarImage, polarLabel = polarConverter.cartesianImageLabel2Polar(image,label,rotation=45)

f3 = plt.figure(frameon=False)
DPI = f3.dpi
H,W = polarImage.shape
rowSubplot= 1
colSubplot= 2
f3.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))
plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(polarImage, cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(polarImage, cmap='gray')
subplot2.plot(polarLabel[0,:,0], polarLabel[0,:,1], linewidth=0.4)
subplot2.plot(polarLabel[1,:,0], polarLabel[1,:,1], linewidth=0.4)
subplot2.axis('off')

plt.savefig(os.path.join(outputDir, patientID + "_polar_rotation.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()


print("========Program Ends=============")

