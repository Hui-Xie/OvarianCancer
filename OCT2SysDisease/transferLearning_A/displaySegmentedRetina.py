
srcDir="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s0tos9_indexSpace/volumes"
srcFile = "359_OD_483_Volume_SegTexture.npy"
outputDir = "/home/hxie1/temp"

k = 15

from PIL import Image  # for Tiff image save
import numpy as np
import os

volumePath = os.path.join(srcDir, srcFile)
volume = np.load(volumePath)
S, H,W = volume.shape

sliceName = srcFile[0:srcFile.find("_SegTexture.npy")] + f"_s{k}.tif"
outputPath = os.path.join(outputDir, sliceName)

print(f"volume.shape = {volume.shape}")

slice = volume[k]

# use PIL to save tiff image, which keep single byte for each pixel.
Image.fromarray(slice).save(outputPath)
