# display raw volume JPG and Segmentation result correspondence for BES3K data.
# for doctors to use

import sys
import os
import glob
import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
from lxml import etree as ET

# Path for original OCT volume
# volumePath = "/home/hxie1/data/BES_3K/raw/34645_OS_29027_Volume"

# Path for OCT-Explorer readable xml segmentation file
# xmlPath  = "/home/hxie1/data/BES_3K/numpy/W512/10SurfPredictionXml/34645_OS_29027_Volume_Sequence_Surfaces_Prediction.xml"

# Path for output directory
# outputDir = "/home/hxie1/data/temp"

# slice index starting from 0
# k = 23  #[0,31) indicate the index of OCT B-scan starting from 0

H=496
K=31

def getSurfacesArray(segFile):
    """
    Read segmentation result into numpy array from OCTExplorer readable xml file.

    :param segFile in xml format
    :return: array in  [Slices, Surfaces,X] order
    """
    xmlTree = ET.parse(segFile)
    xmlTreeRoot = xmlTree.getroot()

    size = xmlTreeRoot.find('scan_characteristics/size')
    for item in size:
        if item.tag == 'x':
            X =int(item.text)
        elif item.tag == 'y':
            Y = int(item.text)
        elif item.tag == 'z':
            Z = int(item.text)
        else:
            continue

    surface_num = int(xmlTreeRoot.find('surface_num').text)
    surfacesArray = np.zeros((Z, surface_num, X),dtype=float)

    s = -1
    for surface in xmlTreeRoot:
        if surface.tag =='surface':
            s += 1
            z = -1
            for bscan in surface:
                if bscan.tag =='bscan':
                   z +=1
                   x = -1
                   for item in bscan:
                       if item.tag =='y':
                           x +=1
                           surfacesArray[z,s,x] = float(item.text)
    return surfacesArray


def printUsage(argv):
    print("\n============ Display segmentation result of BES_3K volume data =============")
    print("Usage:")
    print(argv[0], "<rawVolumePath> <xmlSegFilePath> <outputDir>" )
    print("Explanation of parameters:")
    print("<rawVolumePath> : Path for original OCT volume")
    print("<xmlSegFilePath>: Path for OCT-Explorer readable xml segmentation file")
    print("<outputDir>:      Path for output directory")
    print("Example:")
    print("python3.7 displayJPG_SegXML_BES3K.py /home/hxie1/data/BES_3K/raw/34645_OS_29027_Volume /home/hxie1/data/BES_3K/numpy/W512/10SurfPredictionXml/34645_OS_29027_Volume_Sequence_Surfaces_Prediction.xml /home/hxie1/data/temp")
    print("============ Display segmentation result of BES_3K volume data =============\n")

def main():

    if len(sys.argv) != 4:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    volumePath = sys.argv[1]
    xmlPath = sys.argv[2]
    outputDir = sys.argv[3]
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)  # recursive dir creation

    surfaceNames = ['ILM', 'RNFL-GCL', 'GCL-IPL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE', 'OB_RPE']
    pltColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:olive', 'tab:brown', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:blue']

    imagesList = glob.glob(volumePath + f"/*[0-9][0-9].jpg")
    imagesList.sort()
    if K != len(imagesList):
        print(f"Error: the number of images does not equal {K}. Exit.")
        return
    surfaces = getSurfacesArray(xmlPath)
    Z, S, W = surfaces.shape
    assert K == Z

    print(f"Please wait. It needs 1 minute to output {K} images ......")
    for k in range(K):
        image = imread(imagesList[k])[:, 128:640]
        patientID_Index = os.path.basename(volumePath)+f"_OCT{k+1:02d}"

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
    print(f"===== End: {K} images of {os.path.basename(volumePath)} output at {outputDir} =====")

if __name__ == "__main__":
    main()