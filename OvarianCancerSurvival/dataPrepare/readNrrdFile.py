import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

nrrdPath = "/home/hxie1/data/OvarianCancerCT/rawNrrd/images_1_1_3XYZSpacing/01626917_CT.nrrd"
outputDir = "/home/hxie1/temp"

itkImage = sitk.ReadImage(nrrdPath)
npImage = sitk.GetArrayFromImage(itkImage)

S,H,W = npImage.shape

f = plt.figure(frameon=False)
DPI = f.dpi
rowSubplot= 1
colSubplot= 2
f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
subplot1.imshow(npImage[30,], cmap='gray')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot,colSubplot, 2)
subplot2.imshow(npImage[100,], cmap='gray')
subplot2.axis('off')

plt.savefig(os.path.join(outputDir,"nrrdTest.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()


print(f"==========end======")