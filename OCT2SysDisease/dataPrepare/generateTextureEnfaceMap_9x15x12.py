# generate thickness en-face map

xmlDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"
volumeDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes"
outputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/textureEnfaceMap_9x15x12"
# hPixelSize = 3.870

import glob
import numpy as np
import os
import math
import sys
sys.path.append("../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import getSurfacesArray

stepB=2
stepW=43

xmlVolumeList = glob.glob(xmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
xmlVolumeList.sort()
nXmlVolumes = len(xmlVolumeList)
print(f"total {nXmlVolumes} volumes")

for xmlSegPath in xmlVolumeList:
    basename, ext = os.path.splitext(os.path.basename(xmlSegPath))
    volumeName = basename[0:basename.rfind("_Sequence_Surfaces_Prediction")]
    outputFilename = volumeName + f"_texture_enface" + ".npy"
    outputPath = os.path.join(outputDir, outputFilename)

    # read xml segmentation into array
    volumeSeg = getSurfacesArray(xmlSegPath).astype(np.int)  # BxNxW
    B,N,W = volumeSeg.shape

    # read raw volume
    volumePath = os.path.join(volumeDir, volumeName+".npy")
    volume = np.load(volumePath)  # BxHxW
    _, H, _ = volume.shape


    # define output empty array of size (N-1)xBxW
    textureEnfaceVolume = np.empty((N - 1, B, W), dtype=float)

    # get (N-1)xBxW enface map.
    surface0 = volumeSeg[:, 0:-1, :]  # Bx(N-1)xW
    surface1 = volumeSeg[:, 1:, :]  # Bx(N-1)xW
    thickness = surface1 - surface0  # Bx(N-1)xW   # maybe 0
    for i in range(N - 1):
        for b in range(B):
            for w in range(W):
                textureEnfaceVolume[i, b, w] = volume[b, surface0[b,i, w]: surface1[b,i, w]+1, w].sum() / (thickness[b,i, w]+1)  # BxW

    # reduce image size:
    newB = B // stepB
    newW = math.ceil(W / stepW)
    reducedTextureEnfaceVolume = np.empty((N - 1, newB, newW), dtype=float)
    for i in range(newB):
        for j in range(newW):
            i1 = i*stepB
            i2 = (i+1)*stepB
            if i2 > H:
                i2 = H
            j1 = j*stepW
            j2 = (j+1)*stepW
            if j2 > W:
                j2 = W
            reducedTextureEnfaceVolume[:,i,j] = np.average(textureEnfaceVolume[:,i1:i2,j1:j2], axis=(1,2))


    # output files
    np.save(outputPath, reducedTextureEnfaceVolume)

print(f"=======End of generating texture enface map==========")



