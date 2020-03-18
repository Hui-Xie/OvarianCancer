imagePath1 = "/home/sheen/tempWork/hc02_spectralis_macula_v1_s1_R_21_GT_Predict_NoIPM.png"
imagePath2 = "/home/sheen/tempWork/hc02_spectralis_macula_v1_s1_R_21_GT_Predict_WithIPM.png"
outputDir = "/home/sheen/tempWork"

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

image1 = mpimg.imread(imagePath1)
image2 = mpimg.imread(imagePath2)  # H,W,4


w = 400
h = 0
W = 200
H = 128

gtImage = image1[0:H, w:w+W,:]
badImage = image1[H:2*H, w:w+W,:]
goodImage = image2[H:2*H, w:w+W,:]

f = plt.figure(frameon=False)
DPI = f.dpi
subplotRow = 1
subplotCol = 3
f.set_size_inches(W * subplotCol / float(DPI), H * subplotRow / float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(subplotRow, subplotCol, 1)
subplot1.imshow(gtImage)
subplot1.axis('off')

subplot2 = plt.subplot(subplotRow, subplotCol, 2)
subplot2.imshow(badImage)
subplot2.axis('off')

subplot3 = plt.subplot(subplotRow, subplotCol, 3)
subplot3.imshow(goodImage)
subplot3.axis('off')

plt.savefig(os.path.join(outputDir, "hc02_spectralis_macula_v1_s1_R_21_GT_NoIPM_WithIPM.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()