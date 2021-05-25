
import sys
import SimpleITK as sitk
import os
import glob as glob


def printUsage(argv):
    print("============ convert all nrrd files into mhd/raw files =============")
    print("Usage:")
    print(argv[0], "  NrrdDir")

def convertNrrd2mhd(nrrdPath, mhdPath):
    itkImage = sitk.ReadImage(nrrdPath)
    sitk.WriteImage(itkImage, mhdPath)

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    nrrdDir = sys.argv[1]

    # get files list
    nrrdList = glob.glob(nrrdDir + f"/*.nrrd")
    print(f"start to convert nrrd......\n")
    for nrrdPath in nrrdList:
        basename = os.path.splitext(os.path.basename(nrrdPath))[0]
        mhdPath = os.path.join(nrrdDir, f"{basename}.mhd")
        if os.path.isfile(mhdPath):
            continue
        else:
            convertNrrd2mhd(nrrdPath, mhdPath)
    print(f"all nrrd files have been converted into mhd/raw format in {nrrdDir}.")

if __name__ == "__main__":
    main()