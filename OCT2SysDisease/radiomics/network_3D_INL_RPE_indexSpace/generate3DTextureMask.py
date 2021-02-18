# generate 3D texture and mask from INL to RPE
# region from the surface 3 starting from zero to the surface 8 starting from 0 are the INL_RPE layer.
# save them into nrrd format

textureOutputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/texture3D_nrrd_indexSpace/texture"
maskOutputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/texture3D_nrrd_indexSpace/mask_s3tos8"

srcVolumeDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes"
segXmlDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"

# radiomicsCfgPath = "/home/hxie1/projects/DeepLearningSeg/OCT2SysDisease/radiomics/testConfig/OCTLayerTextureCfg_100Radiomics_3D_indexSpace.yaml"

# surface index starting from zero.
sStart = 3
sEnd = 8

import glob
import numpy as np
#  from PIL import Image  # for Tiff image save.
import SimpleITK as sitk

import sys
sys.path.append("../../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import  getSurfacesArray
import os

def main():
    patientSegsList = glob.glob(segXmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
    print(f"total {len(patientSegsList)} xml files.")
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume
        volumePath = os.path.join(srcVolumeDir, volumeName+".npy")

        maskName = volumeName + f"_s{sStart}tos{sEnd}"

        imagePath = os.path.join(textureOutputDir, volumeName + f"_texture.nrrd")
        maskPath = os.path.join(maskOutputDir, maskName + f"_mask.nrrd")

        volume = np.load(volumePath)  # 31x496x512
        volumeSeg = getSurfacesArray(xmlPath).astype(np.uint32)  # 31x10x512
        S, H, W = volume.shape
        _, N, _ = volumeSeg.shape

        # generate retina layer mask for surface sStart to sEnd
        mask = np.zeros(volume.shape, dtype=np.uint32)  # size: SxHxW
        for s in range(S):
            for c in range(W):
                mask[s, volumeSeg[s,sStart, c]:volumeSeg[s,sEnd, c], c] = 1

        # use PIL to save image for tiff format
        # pyradiomics can not correctly recognize tiff 2D mask
        # Image.fromarray(slice).save(imagePath)
        # Image.fromarray(mask).save(maskPath)

        # use sitk to save image in 3D array
        sitk.WriteImage(sitk.GetImageFromArray(volume), imagePath)
        sitk.WriteImage(sitk.GetImageFromArray(mask), maskPath)

    print(f"===== End of generating texture and mask ==============")


if __name__ == "__main__":
    main()
