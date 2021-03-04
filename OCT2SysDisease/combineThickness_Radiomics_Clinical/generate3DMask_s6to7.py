# generate 3D texture and mask s5 to s7
'''
1 previous analysis:
  the 5th thickness and 6th texture has p-value<0.05 crossing training, validation and test data,
  on the null hypothesis that the mean values in HBP group and non-HBP group are same.
  As this is just a mean value t-test, p-value  of mean value does not capture their real texture features.
2 The gray-level matrix in the radiomics (e.g. gray-level run length matrix) can capture some distance information of similar textures,
  so it can get some thickness information.
3 Logistics regression algoithm has some capabilities to put small weights on the not important features.
4 Therefore, I suggest to use 5th and 6th layers together to extract radiomics information.

'''
# save them into nrrd format

maskOutputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/texture3D_nrrd_indexSpace/mask_s6tos7"
segXmlDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"

# surface index starting from zero.
sStart = 6
sEnd = 7
H = 496


import glob
import numpy as np
#  from PIL import Image  # for Tiff image save.
import SimpleITK as sitk

import sys
sys.path.append("../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import  getSurfacesArray
import os

def main():
    patientSegsList = glob.glob(segXmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
    print(f"total {len(patientSegsList)} xml files.")
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume

        maskName = volumeName + f"_s{sStart}tos{sEnd}"

        maskPath = os.path.join(maskOutputDir, maskName + f"_mask.nrrd")

        volumeSeg = getSurfacesArray(xmlPath).astype(np.uint32)  # 31x10x512
        S, N, W = volumeSeg.shape

        # generate retina layer mask for surface sStart to sEnd
        mask = np.zeros((S,H,W), dtype=np.uint32)  # size: SxHxW
        for s in range(S):
            for c in range(W):
                mask[s, volumeSeg[s,sStart, c]:volumeSeg[s,sEnd, c], c] = 1

        # use PIL to save image for tiff format
        # pyradiomics can not correctly recognize tiff 2D mask
        # Image.fromarray(slice).save(imagePath)
        # Image.fromarray(mask).save(maskPath)

        # use sitk to save image in 3D array
        sitk.WriteImage(sitk.GetImageFromArray(mask), maskPath)

    print(f"===== End of generating mask ==============")


if __name__ == "__main__":
    main()
