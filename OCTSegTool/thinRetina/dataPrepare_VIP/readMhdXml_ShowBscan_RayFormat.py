import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

from utilities import  getSurfacesArray, scaleMatrix, getAllSurfaceNames

import numpy as np

W=200  # target image width
extractIndexs = (0, 1, 3, 5, 6, 10) # extracted surface indexes from original 11 surfaces.

def printUsage(argv):
    print("============ Read Mhd and its XML, and output one Bscan in current directory =============")
    print("Usage:")
    print(argv[0], "  MhdFilePath   SegXmlPathPath   IndexBscan")

def readMhdXml(mhdPath, segXmlPath, indexBscan):
    outputDir = os.getcwd()
    fileName = os.path.splitext(os.path.basename(mhdPath))[0]
    fileName = fileName+f"_Raw_GT_s{indexBscan:03d}.png"
    outputPath = os.path.join(outputDir, fileName)

    itkImage = sitk.ReadImage(mhdPath)
    npImage = sitk.GetArrayFromImage(itkImage).astype(float) # in BxHxW dimension

    # Ray mhd format in BxHxW dimension, but it flip the H and W dimension.
    # for 200x1024x200 image, and 128x1024x512 in BxHxW direction.
    npImage = np.flip(npImage, (1,2)) # as ray's format filp H and W dimension.
    B, H, curW = npImage.shape
    print("\nVolume Information:")
    print(f"Volume name: {mhdPath}")
    print(f"Volume shape in #Bscan x PenetrationDepth x #Ascan format: {npImage.shape}")

    # read segmentation xml file:
    surfaces = getSurfacesArray(segXmlPath)  # size: SxNxW, where N is number of surfacres.
    allSurfaceNames = getAllSurfaceNames(segXmlPath)
    print(f"allSurfaceNames = {allSurfaceNames}")
    surfaces = surfaces[:,extractIndexs,:]
    _, N, _ = surfaces.shape
    print(f"After extraction, surface dimension: {surfaces.shape}")

    if npImage.shape==(128,1024,512): # scale image to 1024x200.
        scaleM = scaleMatrix(B,curW, W)
        npImage = np.matmul(npImage, scaleM)
        surfaces = np.matmul(surfaces, scaleM)
    else:
        assert curW == W

    f = plt.figure(frameon=False)
    DPI = 100
    rowSubplot= 1
    colSubplot= 2
    f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

    subplot1 = plt.subplot(rowSubplot, colSubplot, 1)
    subplot1.imshow(npImage[indexBscan,:,:], cmap='gray')
    subplot1.axis('off')

    subplot2 = plt.subplot(rowSubplot, colSubplot, 2)
    subplot2.imshow(npImage[indexBscan,:,:], cmap='gray')
    for n in range(0, N):
        subplot2.plot(range(0, W), surfaces[indexBscan, n,:], linewidth=0.9)
    subplot2.axis('off')

    plt.savefig(outputPath, dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()
    print("\nOutput:")
    print(f"output the Bscan {indexBscan} with surface at: {outputPath}\n")

def main():
    if len(sys.argv) != 4:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    readMhdXml(sys.argv[1], sys.argv[2], int(sys.argv[3]))

if __name__ == "__main__":
    main()