# generate thickness en-face map

xmlDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"
outputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thicknessEnfaceMap"
hPixelSize = 3.870

import glob
import numpy as np
import os
import sys
sys.path.append("../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import getSurfacesArray

xmlVolumeList = glob.glob(xmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
xmlVolumeList.sort()
nXmlVolumes = len(xmlVolumeList)
print(f"total {nXmlVolumes} volumes")

for xmlSegPath in xmlVolumeList:
    basename, ext = os.path.splitext(os.path.basename(xmlSegPath))
    outputFilename = basename[0:basename.rfind("_Sequence_Surfaces_Prediction")] + f"_thickness_Enface" + ".npy"
    outputPath = os.path.join(outputDir, outputFilename)

    # read xml segmentation into array
    volumeSeg = getSurfacesArray(xmlSegPath).astype(np.int)  # BxNxW
    B,N,W = volumeSeg.shape

    thicknessEnface = np.empty((N - 1, B, W), dtype=np.float)
    # fill the output array
    for i in range(N - 1):
        surface0 = volumeSeg[:, i, :]  # BxW
        surface1 = volumeSeg[:, i + 1, :]  # BxW
        thickness = surface1 - surface0  # BxW # maybe 0
        thicknessEnface[i, :, :] = thickness * hPixelSize
        # do 3*3 mean filter on BxW dimension



    # output files
    np.save(outputPath, thicknessEnface)

    break

print(f"=======End of generating thickness enface map==========")



