import sys
from DataMgr import DataMgr
import numpy as np


def main():
    if len(sys.argv) != 3:
        print("Error: input parameters error.")
        return -1

    dataMgr = DataMgr(sys.argv[1], sys.argv[2])
    filesList = dataMgr.getFilesList(dataMgr.m_imagesDir, "_CT.nrrd")
    dataMgr.setDataSize(4, 21,281,281,4)
    #imageFile = filesList[0]
    imageFile = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages/05739688_CT.nrrd"
    labelFile = dataMgr.getLabelFile(imageFile)

    print("labelFile: ", labelFile)
    labelArray = dataMgr.readImageFile(labelFile)
    sliceIndex = dataMgr.getLabeledSliceIndex(labelArray)
    print(sliceIndex)

    print(imageFile)
    imageArray = dataMgr.readImageFile(imageFile)
    dataMgr.displaySlices(imageArray.clip(-400, 500), sliceIndex)
    dataMgr.displaySlices(labelArray, sliceIndex)

    dataMgr.readImageAttributes(imageFile)

    dataMgr.saveImage(imageArray, [0,0,0], "/home/hxie1/temp/testFullImage.nrrd")

    partImageArray = imageArray[20:200, 30:300, 40:250];
    dataMgr.saveImage(partImageArray, [20, 30, 40], "/home/hxie1/temp/testPartImage.nrrd")

    for i in range(len(sliceIndex)):
        oneSliceSeg = labelArray[sliceIndex[i]]
        oneHotArray = dataMgr.segmentation2OneHotArray(oneSliceSeg, 4)
        reconstructOneSliceSeg = dataMgr.oneHotArray2Segmentation(oneHotArray)
        if np.array_equal(oneSliceSeg, reconstructOneSliceSeg):
            print(f'Good at {i} seg slice, as its reconstrunct slice is same with original segSlice')
        else:
            print(f'oneSliceSeg has {np.count_nonzero(oneSliceSeg)} nonzeros')
            print(f'reconstructOneSliceSeg has {np.count_nonzero(reconstructOneSliceSeg)} nonzeros')
            print(f'Bad at {i} seg slice, as its reconstrunct slice is NOT same with original segSlice')


    # test dataLabelGenerator
    #generator = dataMgr.dataLabelGenerator(False)
    #for data, label in generator:
    #    print(f'data shape: {data.shape}')
    #    print(f'label shape: {label.shape}')


    dataMgr.checkOrientConsistent("/home/hxie1/data/R21Project/CTSegV_uniform/testImages", "_CT.nrrd")


    print("=============END=================")



if __name__ == "__main__":
    main()
