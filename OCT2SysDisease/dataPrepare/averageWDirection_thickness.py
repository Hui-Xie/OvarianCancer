# average width direction
'''
240.55/11.29 = 21.3, at X direction, average 21 pixels into 1, so orignal enface image becomes 31x24.38 = 31x25 pixels.
It guarantees that pixel has same resolution at Z and X direction. 512 = 24x21+1x8

'''

import glob
import os
import numpy as np
import math

srcDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thicknessEnfaceMap_9x31x512"
dstDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thicknessEnfaceMap_9x31x25"

fileSuffix = "_Volume_thickness_enface.npy"

W = 512 # original image x dimension
d = 21 # average number of pixels in width direction
newW = math.ceil(W/d)

srcVolumeList = glob.glob(srcDir + f"/*{fileSuffix}")
srcVolumeList.sort()
nSrcVolumes = len(srcVolumeList)
print(f"total {nSrcVolumes} volumes")


for volumePath in srcVolumeList:
    outputFilename = os.path.basename(volumePath)
    outputPath = os.path.join(dstDir, outputFilename)

    # read volume
    volume = np.load(volumePath)  # BxHxW
    B, H, W1 = volume.shape
    assert B==9 and H==31 and W1==W

    newVolume = np.empty((B,H,newW),dtype=float)
    for i in range(newW):
        j1 = i*d
        j2 = (i+1)*d
        if j2 > W:
            j2 = W
        newVolume[:,:,i] = np.average(volume[:,:,j1:j2], axis=2)
    np.save(outputPath,newVolume)
print(f"=============End of average width direction ===============")
