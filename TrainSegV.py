import sys
from DataMgr import DataMgr


def main():
    if len(sys.argv) != 3:
        print("Error: input parameters error.")
        return -1

    dataMgr = DataMgr(sys.argv[1], sys.argv[2])
    filesList = dataMgr.getFilesList(dataMgr.m_imagesDir, "_CT.nrrd")
    #imageFile = filesList[0]
    imageFile = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/images/01626917_CT.nrrd"



    labelFile = dataMgr.getLabelFile(imageFile)
    print("labelFile: ", labelFile)
    labelArray = dataMgr.readImageFile(labelFile)
    sliceIndex = dataMgr.getLabeledSliceIndex(labelArray)
    print(sliceIndex)

    print(imageFile)
    imageArray = dataMgr.readImageFile(imageFile)
    dataMgr.displaySlices(imageArray.clip(-400, 500), sliceIndex)
    dataMgr.displaySlices(labelArray, sliceIndex)




if __name__ == "__main__":
    main()
