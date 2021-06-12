# generate 5th thickness as channel 0 and 6th texture as channel 1

outputDir= "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/5thThickness_6thTexture_2x15x12"
thicknessDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thicknessEnfaceMap_9x15x12"
textureDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/textureEnfaceMap_9x15x12"


import glob
import os
import numpy as np

c1=5 # 5th thickness
c2=6 # 6th texture

fileSuffix = "_Volume_thickness_enface.npy"

thicknessVolumeList = glob.glob(thicknessDir + f"/*{fileSuffix}")
thicknessVolumeList.sort()
nVolumes = len(thicknessVolumeList)
print(f"total {nVolumes} volumes")

if not os.path.exists(outputDir):
    os.makedirs(outputDir)  # recursive dir creation

for thickessPath in thicknessVolumeList:
    texturePath = thickessPath.replace("thickness", "texture")

    outputFilename = os.path.basename(thickessPath)
    outputFilename = outputFilename.replace("_thickness_", "_5thThickness_6thTexture_")
    outputPath = os.path.join(outputDir, outputFilename)

    newVolume = np.empty((2,15,12), dtype=float)
    # read volume
    thicknessVolume = np.load(thickessPath)  # BxHxW
    assert (9,15,12) == thicknessVolume.shape

    textureVolume = np.load(texturePath)
    assert (9,15, 12) == textureVolume.shape

    newVolume[0,] = thicknessVolume[c1,]
    newVolume[1,] = textureVolume[c2,]
    np.save(outputPath,newVolume)

print(f"=============End of generate {c1}th thickness and {c2}th texture enface map===============")
