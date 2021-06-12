# generate thickness en-face map with size 9x15x12

xmlDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"
outputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thicknessEnfaceMap_9x15x12"
hPixelSize = 3.870

import glob
import numpy as np
import os
import sys
import math
sys.path.append("../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import getSurfacesArray

xmlVolumeList = glob.glob(xmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
xmlVolumeList.sort()
nXmlVolumes = len(xmlVolumeList)
print(f"total {nXmlVolumes} volumes")

stepH=2
stepW=43

for xmlSegPath in xmlVolumeList:
    basename, ext = os.path.splitext(os.path.basename(xmlSegPath))
    outputFilename = basename[0:basename.rfind("_Sequence_Surfaces_Prediction")] + f"_thickness_enface" + ".npy"
    outputPath = os.path.join(outputDir, outputFilename)

    # read xml segmentation into array
    volumeSeg = getSurfacesArray(xmlSegPath).astype(np.int)  # BxNxW
    H,N,W = volumeSeg.shape

    newH = H//stepH
    newW = math.ceil(W/stepW)


    thicknessEnface = np.empty((N - 1, newH, newW), dtype=float)

    surface0 = volumeSeg[:, 0:-1, :]  # Bx(N-1)xW
    surface1 = volumeSeg[:, 1:, :]  # Bx(N-1)xW
    thickness = (surface1 - surface0).astype(float)  # Bx(N-1)xW
    thickness = np.swapaxes(thickness, 0, 1)  # (N-1)xBxW

    for i in range(newH):
        for j in range(newW):
            i1 = i*stepH
            i2 = (i+1)*stepH
            if i2 > H:
                i2 = H
            j1 = j*stepW
            j2 = (j+1)*stepW
            if j2 > W:
                j2 = W
            thicknessEnface[:,i,j] = np.average(thickness[:,i1:i2,j1:j2], axis=(1,2))

    thicknessEnface = thicknessEnface * hPixelSize

    # output files
    np.save(outputPath, thicknessEnface)

print(f"=======End of generating thickness enface map 9x15x12 ==========")



