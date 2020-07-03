# check raw volume JPG and Segmentation result correspondence for BES3K data.

import os
import glob
import matplotlib.pyplot as plt
from TongrenFileUtilies import getSurfacesArray
from imageio import imread



# Path for original OCT volume
volumePath = "/home/hxie1/data/BES_3K/raw/34645_OS_29027_Volume"
# Path for OCT-Explorer readable xml segmentation file
xmlPath  = "/home/hxie1/data/BES_3K/numpy/W512/10SurfPredictionXml/34645_OS_29027_Volume_Sequence_Surfaces_Prediction.xml"
# Path for output directory
outputDir = "/home/hxie1/data/temp"
# slice index starting from 0
k = 23  #[0,31) indicate the index of OCT B-scan starting from 0

H=496


def main():
    imagesList = glob.glob(volumePath + f"/*[0-9][0-9].jpg")
    imagesList.sort()
    image = imread(imagesList[k])[:, 128:640]

    patientID_Index = os.path.basename(volumePath)+f"_OCT{k+1:02d}"

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