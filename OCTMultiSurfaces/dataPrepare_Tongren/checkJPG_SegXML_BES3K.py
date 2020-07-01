# check raw volume JPG and Segmentation result correspondence for BES3K data.


import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from TongrenFileUtilities import *
from imageio import imread

volumePath = "/home/hxie1/data/BES_3K/raw/106_OS_5451_Volume"
xmlPath  = "/home/hxie1/data/BES_3K/numpy/W512/log/SurfacesUnet/expBES3K_10Surfaces_20200701_test_GPU2/testResult/xml/106_OS_5451_Volume_Sequence_Surfaces_Prediction.xml"
outputDir = "/home/hxie1/temp"
k = 15  #0-31 indicate the OCT B-scan
H=496


def main():
    imagesList = glob.glob(volumePath + f"/*[0-9][0-9].jpg")
    imagesList.sort()
    image = imread(imagesList[k])[:, 128:640]

    patientID_Index = os.path.basename(volumePath)+f"_OCT{k+1}"

    surfaces = getSurfacesArray(xmlPath)
    Z, S, W = surfaces.shape

    surfaceNames = ['ILM', 'RNFL-GCL', 'GCL-IPL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE','OB_RPE']
    pltColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:olive', 'tab:brown', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:blue']

    f = plt.figure(frameon=False)
    DPI = f.dpi
    rowSubplot = 1
    colSubplot = 2
    f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot / float(DPI))
    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.


    subplot1 = plt.subplot(rowSubplot, colSubplot, 1)
    subplot1.imshow(image, cmap='gray')
    subplot1.axis('off')

    subplot2 = plt.subplot(rowSubplot, colSubplot, 2)
    subplot2.imshow(image, cmap='gray')
    for s in range(0, S):
        subplot2.plot(range(0, W), surfaces[k, s, :],  pltColors[s], linewidth=0.9)
    subplot2.legend(surfaceNames, loc='lower center', ncol=4)
    subplot2.axis('off')

    plt.savefig(os.path.join(outputDir, patientID_Index + "_Raw_Prediction.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()
    print("============End of Program=============")


if __name__ == "__main__":
    main()