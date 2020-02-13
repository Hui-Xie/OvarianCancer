# verify polar image image and label

imagePath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/DCM/frame_02_0005_003.png"
lumenLabelPath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS/lum_frame_02_0005_003.txt"
mediaLabelPath = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS/med_frame_02_0005_003.txt"
outputDir = "/home/hxie1/temp"


import sys
sys.path.append(".")
from PolarCartesianConverter import PolarCartesianConverter




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

# verify the ground truth order
subplot3 = plt.subplot(rowSubplot,colSubplot, 3)
subplot3.imshow(image, cmap='gray')
for i in range(0, 360, 45):
    subplot3.text(lumenLabel[i,0], lumenLabel[i,1], str(i//45), color = 'blue')
    subplot3.text(mediaLabel[i,0], mediaLabel[i,1], str(i//45), color = 'red')
subplot3.axis('off')

plt.savefig(os.path.join(outputDir, patientID + "_Raw_GT.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()


# convert polar image
polarConverter = PolarCartesianConverter(image.shape, W//2,H//2,min(W//2,H//2), 360)
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
#polarImage, polarLabel = polarConverter.cartesianImageLabel2Polar(image,label,rotation=45)
# use 2 lines:
polarImage1, polarLabel1 = polarConverter.cartesianImageLabel2Polar(image,label,rotation=0)
polarImage2, polarLabel2 = polarConverter.polarImageLabelRotate(polarImage1, polarLabel1, rotation=60)

f3 = plt.figure(frameon=False)
DPI = f3.dpi
H,W = polarImage.shape
rowSubplot= 2
colSubplot= 2
f3.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))
plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(polarImage1, cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(polarImage1, cmap='gray')
subplot2.plot(polarLabel1[0,:,0], polarLabel1[0,:,1], linewidth=0.4)
subplot2.plot(polarLabel1[1,:,0],  polarLabel1[1,:,1], linewidth=0.4)
subplot2.axis('off')

subplot3 = plt.subplot(rowSubplot,colSubplot, 3)
subplot3.imshow(polarImage2, cmap='gray')
subplot3.axis('off')

subplot4 = plt.subplot(rowSubplot,colSubplot, 4)
subplot4.imshow(polarImage2, cmap='gray')
subplot4.plot(polarLabel2[0,:,0], polarLabel2[0,:,1], linewidth=0.4)
subplot4.plot(polarLabel2[1,:,0], polarLabel2[1,:,1], linewidth=0.4)
subplot4.axis('off')

plt.savefig(os.path.join(outputDir, patientID + "_polar_rotation.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()


# convert polar image  back to cartesian image
cartesianImage, cartesianLabel = polarConverter.polarImageLabel2Cartesian(polarImage,polarLabel,rotation=45)

f4 = plt.figure(frameon=False)
DPI = f4.dpi
H,W = cartesianImage.shape
rowSubplot= 1
colSubplot= 3
f4.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))
plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(cartesianImage, cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(cartesianImage, cmap='gray')
subplot2.plot(cartesianLabel[0,:,0], cartesianLabel[0,:,1], linewidth=0.4)
subplot2.plot(cartesianLabel[1,:,0], cartesianLabel[1,:,1], linewidth=0.4)
subplot2.axis('off')

# verify the ground truth order
subplot3 = plt.subplot(rowSubplot,colSubplot, 3)
subplot3.imshow(cartesianImage, cmap='gray')
for i in range(0, 360, 45):
    subplot3.text(cartesianLabel[0,i,0], cartesianLabel[0,i,1], str(i//45), color = 'blue')
    subplot3.text(cartesianLabel[1,i,0], cartesianLabel[1,i,1], str(i//45), color = 'red')
subplot3.axis('off')

plt.savefig(os.path.join(outputDir, patientID + "_polar_rotation_backCartesian.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()



print("========Program Ends=============")

