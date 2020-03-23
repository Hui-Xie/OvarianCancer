# check JPG and Segmentation result correspondence for glaucoma data.


import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from TongrenFileUtilities import *

imagePath = "/home/hxie1/data/OCT_Tongren/glaucomaImages_W512/4089_OD_30949_Volume/16.jpg"
xmlPath  = "/home/hxie1/data/OCT_Tongren/numpy/glaucomaRaw_W512/log/SurfacesUnet/expUnetTongren_20200323_Glaucoma_9Surfaces_test/testResult/xml/4089_OD_30949_Volume_Volume_Sequence_Surfaces_Prediction.xml"
outputDir = "/home/hxie1/temp"
z = 15 # 16-1
H=496


def main():
    surfacesArray = getSurfacesArray(xmlPath)
    Z,surface_num, W = surfacesArray.shape



    f = plt.figure(frameon=False)
    DPI = f.dpi
    f.set_size_inches(W/ float(DPI), H / float(DPI))
    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

    image = mpimg.imread(imagePath)
    plt.imshow(image, cmap='gray')
    for s in range(0, surface_num):
        plt.plot(range(0,W), surfacesArray[z,s,:], linewidth=0.9)
    plt.axis('off')

    plt.savefig(os.path.join(outputDir, "readGlaucomaXML.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()
    print("============End of Program=============")


if __name__ == "__main__":
    main()