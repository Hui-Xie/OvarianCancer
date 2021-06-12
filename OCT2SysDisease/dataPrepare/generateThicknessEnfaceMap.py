# generate thickness en-face map

xmlDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"
outputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thicknessEnfaceMap_9x31x512"
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

#kernel = np.ones((3, 3), float) / 9.0 # for 2D smooth filter

for xmlSegPath in xmlVolumeList:
    basename, ext = os.path.splitext(os.path.basename(xmlSegPath))
    outputFilename = basename[0:basename.rfind("_Sequence_Surfaces_Prediction")] + f"_thickness_enface" + ".npy"
    outputPath = os.path.join(outputDir, outputFilename)

    # read xml segmentation into array
    volumeSeg = getSurfacesArray(xmlSegPath).astype(np.int)  # BxNxW
    B,N,W = volumeSeg.shape

    surface0 = volumeSeg[:, 0:-1, :]  # Bx(N-1)xW
    surface1 = volumeSeg[:, 1:, :]  # Bx(N-1)xW
    thickness = (surface1 - surface0).astype(float)  # Bx(N-1)xW
    thickness = np.swapaxes(thickness, 0, 1)  # (N-1)xBxW
    # do 3*3 mean filter on BxW dimension
    # because of the Z and X has big pixel resolution difference, cancel smooth.
    # thickness = cv. filter2D(thickness,-1,kernel, borderType=cv.BORDER_REPLICATE)
    thicknessEnface = thickness * hPixelSize

    # output files
    np.save(outputPath, thicknessEnface)

print(f"=======End of generating thickness enface map==========")



