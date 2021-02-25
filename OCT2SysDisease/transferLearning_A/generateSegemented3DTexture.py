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


'''
segmented images mean and std, which is already normalized in the output of output:
/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s0tos9_indexSpace/volumes

(base) [c-xwu000:volume3D_s0tos9_indexSpace]#cat outputLog_20210224_174528.txt
total 6499 xml files.
output segmented and normalized retina regions from surface 0 to surface9.
Now all masks have been generated in /home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s0tos9_indexSpace/masks
meanBscans = [0.05199834 0.05284249 0.0532695  0.0537375  0.05438357 0.0549899
 0.0558392  0.05696388 0.05825874 0.0598245  0.06093201 0.06143699
 0.06085205 0.05910843 0.05703062 0.05591611 0.05711094 0.05924026
 0.06057586 0.06111015 0.06090984 0.0599876  0.05905826 0.05801464
 0.05705912 0.056168   0.05544387 0.05496442 0.05464729 0.05429021
 0.0540758 ]
stdBscans = [0.14800051 0.14917299 0.14980883 0.1504818  0.15139771 0.15180002
 0.15233731 0.15314984 0.15386566 0.15500712 0.15508947 0.15453244
 0.15276628 0.15002041 0.14744838 0.14581831 0.14734901 0.14996752
 0.15186637 0.15354847 0.15457117 0.15440928 0.15420679 0.15358165
 0.15280802 0.15198185 0.15122363 0.1508052  0.15057934 0.15009072
 0.14986088]
'''

import os

outputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s0tos9_indexSpace"
textureOutputDir = os.path.join(outputDir,"volumes")
maskOutputDir = os.path.join(outputDir,"masks")


srcVolumeDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes"
segXmlDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"

# surface index starting from zero.
sStart = 0
sEnd = 9   # the whole segmented section.

import glob
import numpy as np

import sys
sys.path.append("../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import  getSurfacesArray

import datetime

output2File = True

def main():
    S = 31
    H = 496
    W = 512
    N = 10  # surface number

    # prepare output file
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(outputDir, f"outputLog_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    patientSegsList = glob.glob(segXmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
    print(f"total {len(patientSegsList)} xml files.")
    print(f"output segmented and normalized retina regions from surface {sStart} to surface{sEnd}.")

    # first generate masks
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume
        maskPath = os.path.join(maskOutputDir, f"{volumeName}_mask.npy")

        if os.path.isfile(maskPath):  # generating one file needs 30 seconds.
            continue

        volumeSeg = getSurfacesArray(xmlPath).astype(np.uint32)  # 31x10x512

        # generate retina layer mask for surface sStart to sEnd
        mask = np.zeros((S,H,W), dtype=np.uint32)  # size: SxHxW
        for s in range(S):
            for c in range(W):
                mask[s, volumeSeg[s, sStart, c]:volumeSeg[s, sEnd, c], c] = 1

        np.save(maskPath, mask)
    print(f"Now all masks have been generated in {maskOutputDir}")


    # first compute mean for each B-scan
    nPatients = 0
    sumBscans = np.zeros((S), dtype=np.float)
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume
        volumePath = os.path.join(srcVolumeDir, volumeName+".npy")
        maskPath = os.path.join(maskOutputDir, f"{volumeName}_mask.npy")

        volume = np.load(volumePath)  # 31x496x512
        mask = np.load(maskPath)

        retinaVolume = (volume * mask)/255.0  # size: 31x496x512
        nPatients += 1
        sumBscans =sumBscans + retinaVolume.sum(axis=(1,2))
    meanBscans = sumBscans/(nPatients*H*W)
    print(f"meanBscans = {meanBscans}")

    # compute std for each B-scan
    nPatients = 0
    sumBscans = np.zeros((S), dtype=np.float)
    meanBscansExpand = np.tile(meanBscans.reshape(S,1,1), reps=(1,H,W))
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume
        volumePath = os.path.join(srcVolumeDir, volumeName + ".npy")

        maskPath = os.path.join(maskOutputDir, f"{volumeName}_mask.npy")
        volume = np.load(volumePath)  # 31x496x512
        mask = np.load(maskPath)

        retinaVolume = (volume * mask) / 255.0  # size: 31x496x512
        nPatients += 1

        sumBscans = sumBscans + np.square(retinaVolume-meanBscansExpand).sum(axis=(1, 2))
    stdBscans = np.sqrt(sumBscans / (nPatients * H * W))
    print(f"stdBscans = {stdBscans}")

    # normalization and save to numpy file
    stdBscansExpand = np.tile(stdBscans.reshape(S, 1, 1), reps=(1, H, W))
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume
        volumePath = os.path.join(srcVolumeDir, volumeName + ".npy")

        imagePath = os.path.join(textureOutputDir, volumeName + f"_SegTexture.npy")
        if os.path.isfile(imagePath):  # generating one file needs 30 seconds.
            continue

        maskPath = os.path.join(maskOutputDir, f"{volumeName}_mask.npy")
        volume = np.load(volumePath)  # 31x496x512
        mask = np.load(maskPath)

        retinaVolume = (volume * mask) / 255.0  # size: 31x496x512
        normVolume = (retinaVolume-meanBscansExpand)/stdBscansExpand

        np.save(imagePath, normVolume)

    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"===== End of generating 3D texture ==============")



if __name__ == "__main__":
    main()
