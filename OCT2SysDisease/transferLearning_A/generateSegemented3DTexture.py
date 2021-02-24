# generate segmented 3D texture
#   original volume pixel value in [0, 255],
#   All pre-trained models expect input images normalized in the same way,
#   i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.
#   The images have to be loaded in to a range of [0, 1]
#   and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
#  for OCT image:
#  A.  Input: 31x496x512 in pixel space, where 31 is the number of B-scans.
#  B.  Use segmented mask to exclude non-retina region, e.g. vitreous body and choroid etc, getting 31x496x512 filtered volume.
#  C.  image /255
#  D.  We should compute the mean and std of each channels of 31 channels of OCT images.
#  E.  normalization: (image-mean)/std.
#  F.  save as npy file.


textureOutputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s0tos9_indexSpace"

srcVolumeDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes"
segXmlDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"

# surface index starting from zero.
sStart = 0
sEnd = 9   # the whole segmented section.

import glob
import numpy as np
#  from PIL import Image  # for Tiff image save.
import SimpleITK as sitk

import sys
sys.path.append("../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import  getSurfacesArray
import os
import datetime

output2File = True
S = 31
H = 496
W = 512
N = 10

def main():
    # prepare output file
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(textureOutputDir, f"outputLog_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    patientSegsList = glob.glob(segXmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
    print(f"total {len(patientSegsList)} xml files.")

    # first compute mean for each B-scan
    nPatients = 0
    sumBscans = np.zeros((32), dtype=np.float)
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume
        volumePath = os.path.join(srcVolumeDir, volumeName+".npy")

        volume = np.load(volumePath)  # 31x496x512
        volumeSeg = getSurfacesArray(xmlPath).astype(np.uint32)  # 31x10x512

        # generate retina layer mask for surface sStart to sEnd
        mask = np.zeros(volume.shape, dtype=np.uint32)  # size: SxHxW
        for s in range(S):
            for c in range(W):
                mask[s, volumeSeg[s,sStart, c]:volumeSeg[s,sEnd, c], c] = 1

        retinaVolume = (volume * mask)/255.0  # size: 31x496x512
        nPatients += 1
        sumBscans =sumBscans + retinaVolume.sum(axis=[1,2])
    meanBscans = sumBscans/(nPatients*H*W)
    print(f"meanBscans = {meanBscans}")

    return






    # compute std

    # normalization and save to numpy file
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


    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"===== End of generating 3D texture ==============")



if __name__ == "__main__":
    main()
