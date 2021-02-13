# generate texture and mask for specific slice
# save them into tiff format

textureOutputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15TextureMask/texture"
maskOutputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15TextureMask/mask"

srcVolumeDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes"
segXmlDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"

radiomicsCfgPath = "/home/hxie1/projects/DeepLearningSeg/OCT2SysDisease/radiomics/testConfig/OCTLayerTextureCfg_95Radiomics.yaml"
indexBscan = 15

import glob
import numpy as np
from PIL import Image

import sys
sys.path.append("../../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import  getSurfacesArray
import os

import radiomics
from radiomics import featureextractor, getFeatureClasses


def main():
    patientSegsList = glob.glob(segXmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
    print(f"total {len(patientSegsList)} xml files.")
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume
        volumePath = os.path.join(srcVolumeDir, volumeName+".npy")

        sliceName = volumeName + f"_s{indexBscan}"

        imagePath = os.path.join(textureOutputDir, sliceName + f"_texture.tif")  # It must use tiff format for float format
        maskPath = os.path.join(maskOutputDir, sliceName + f"_mask.tif")

        volume = np.load(volumePath)  # 31x496x512
        volumeSeg = getSurfacesArray(xmlPath).astype(np.uint32)  # 31x10x512
        slice = volume[indexBscan,]  # 496x512
        H, W = slice.shape
        sliceSeg = volumeSeg[indexBscan,]  # 10x512
        N, W = sliceSeg.shape

        # generate retina layer mask
        mask = np.zeros(slice.shape, dtype=np.uint32)  # size: HxW
        for c in range(W):
            mask[sliceSeg[0, c]:sliceSeg[N - 1, c], c] = 1

        # use PIL to save image
        Image.fromarray(slice).save(imagePath)
        Image.fromarray(mask).save(maskPath)


if __name__ == "__main__":
    main()
