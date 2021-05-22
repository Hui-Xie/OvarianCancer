import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os

def printUsage(argv):
    print("============ Read Nrrd and output one Bscan in current directory =============")
    print("Usage:")
    print(argv[0], "  NrrdFilePath   IndexBscan")


def readNrrd(nrrdPath, indexBscan):
    outputDir = os.getcwd()
    fileName = os.path.splitext(os.path.basename(nrrdPath))[0]
    fileName = fileName+f"_{indexBscan:03d}.png"
    outputPath = os.path.join(outputDir, fileName)

    itkImage = sitk.ReadImage(nrrdPath)
    npImage = sitk.GetArrayFromImage(itkImage)

    S,H,W = npImage.shape
    print(f"{nrrdPath} has a shape: {npImage.shape} in #Bscan x PenetrationDepth x #Ascan format.")

    f = plt.figure(frameon=False)
    DPI = 100
    rowSubplot= 1
    colSubplot= 1
    f.set_size_inches(W*colSubplot/ float(DPI), H*rowSubplot/ float(DPI))

    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

    subplot1 = plt.subplot(rowSubplot,colSubplot, 1)
    subplot1.imshow(npImage[indexBscan,], cmap='gray')
    subplot1.axis('off')

    plt.savefig(outputPath, dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"output Bscan {indexBscan} at: {outputPath}")

def main():
    if len(sys.argv) != 3:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    readNrrd(sys.argv[1], int(sys.argv[2]))



if __name__ == "__main__":
    main()