# generate 5th thickness as channel 0

outputDir= "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/5thThickness_1x15x12"
thicknessDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thicknessEnfaceMap_9x15x12"

import glob
import os
import numpy as np

c1=5 # 5th thickness

fileSuffix = "_Volume_thickness_enface.npy"

thicknessVolumeList = glob.glob(thicknessDir + f"/*{fileSuffix}")
thicknessVolumeList.sort()
nVolumes = len(thicknessVolumeList)
print(f"total {nVolumes} volumes")

if not os.path.exists(outputDir):
    os.makedirs(outputDir)  # recursive dir creation

for thickessPath in thicknessVolumeList:
    outputFilename = os.path.basename(thickessPath)
    outputPath = os.path.join(outputDir, outputFilename)

    newVolume = np.empty((1,15,12), dtype=float)
    # read volume
    thicknessVolume = np.load(thickessPath)  # BxHxW
    assert (9,15,12) == thicknessVolume.shape

    newVolume[0,] = thicknessVolume[c1,]
    np.save(outputPath,newVolume)

print(f"=============End of generate {c1}th thickness map at {outputDir} ===============")
