# Extract segmented OCT layer texture features.

srcVolumePath ="/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes/2081_OD_14279_Volume.npy"
segXmlPath ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml/2081_OD_14279_Volume_Sequence_Surfaces_Prediction.xml"
outputDir = "/home/hxie1/temp/extractRadiomics"

radiomicsCfgPath = "/home/sheen/projects/DeepLearningSeg/OCT2SysDisease/radiomics/testConfig/OCTLayerTextureCfg.yaml"
indexBscan = 15


import numpy as np
import os
import imageio

import sys
sys.path.append("../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import  getSurfacesArray


def generateImage_Mask(volumePath, xmlPath, indexBscan, outputDir):
    volumeName, _ = os.path.splitext(os.path.basename(volumePath))
    sliceName = volumeName + f"_s{indexBscan}"

    imagePath = os.path.join(outputDir, sliceName + f"_texture.jpg")
    maskPath = os.path.join(outputDir, sliceName + f"_mask.jpg")

    volume = np.load(volumePath)  # 31x496x512
    volumeSeg  = getSurfacesArray(xmlPath)  # 31x10x512
    slice = volume[indexBscan,]  # 496x512
    H,W = slice.shape
    sliceSeg = volumeSeg[indexBscan,]  # 10x512
    N,W = sliceSeg.shape

    #generate retina layer mask
    mask = np.zeros(slice.shape, dtype=np.int)  # size: HxW
    for c in range(W):
        mask[sliceSeg[0,c]:sliceSeg[N-1,c],c] = 1

    # save slice and mask
    imageio.imwrite(imagePath,slice)
    imageio.imwrite(maskPath, mask)

    return imagePath, maskPath

def generateRadiomics(imagePath, maskPath, radiomicsCfgPath):
    pass


def main():
    imagePath, maskPath = generateImage_Mask(srcVolumePath, segXmlPath, indexBscan, outputDir)
    generateRadiomics(imagePath,maskPath)


if __name__ == "__main__":
    main()
