# generate en-face images and layerWidth file from segmentation result and OCT volume

import glob
import numpy as np
import os
import sys
sys.path.append("..")
from dataPrepare_Tongren.TongrenFileUtilities import getSurfacesArray


# imageIDPath = "/home/hxie1/data/BES_3K/GTs/allID_delNonExist_delErrWID_excludeMGM.csv"
'''
# read volumeID into list
with open(imageIDPath, 'r') as idFile:
    IDList = idFile.readlines()
IDList = [item[0:-1] for item in IDList]  # erase '\n'

'''

hPixelSize =  3.870  # unit: micrometer, in y/height direction
OCTVolumeDir = "/home/hxie1/data/BES_3K/numpy_10SurfaceSeg/W512/testVolume"  # in npy, a file per volume
segXmlDir ="/home/hxie1/data/BES_3K/numpy_10SurfaceSeg/W512/10SurfPredictionXml"
outputDirEnface ="/home/hxie1/data/BES_3K/numpy_10SurfaceSeg/W512/enfaceW512"
outputDirWidth ="/home/hxie1/data/BES_3K/numpy_10SurfaceSeg/W512/layerWidthW512"

volumesList = glob.glob(OCTVolumeDir + f"/*_Volume.npy")
for volumePath in volumesList:
    # read volume
    volume = np.load(volumePath)  # BxHxW
    B, H, W = volume.shape

    basename = os.path.basename(volumePath)
    volumename, ext = os.path.splitext(basename)
    xmlSegName = volumename+ "_Sequence_Surfaces_Prediction.xml"
    xmlSegPath = os.path.join(segXmlDir, xmlSegName)
    if not os.path.exists(xmlSegPath):
        print(f"file not exist: {xmlSegPath}")
        continue

    # read xml segmentation into array
    volumeSeg = getSurfacesArray(xmlSegPath).astype(np.int) # BxNxW
    B1, N, W1 = volumeSeg.shape
    assert (B == B1) and (W == W1)

    # define output emtpy array
    enfaceVolume = np.empty((N-1, B, W), dtype=np.float)
    layerWidthVolume = np.empty((N-1, B, W), dtype=np.float)
    # fill the output array
    for i in range(N-1):
        surface0 = volumeSeg[:,i,:]  # BxW
        surface1 = volumeSeg[:,i+1,:]  # BxW
        width = surface1 - surface0  # BxW # maybe 0
        for b in range(B):
            for w in range(W):
                if 0 == width[b,w]:
                    enfaceVolume[i, b, w] = volume[b,surface0[b,w], w]
                elif width[b,w] >= 1:
                    enfaceVolume[i, b, w] = volume[b,surface0[b,w]: surface1[b,w], w].sum()/width[b,w]  #BxW
                else:
                    print(f"Error: layer width is negative at b={b} and w={w} in {volumePath}")

        layerWidthVolume[i,:,:] = width *hPixelSize


    # output file
    enFaceVolumePath = os.path.join(outputDirEnface, f"{volumename}_retina{N-1}Layers_enface.npy")
    np.save(enFaceVolumePath, enfaceVolume)

    layerWidthVolumePath = os.path.join(outputDirWidth, f"{volumename}_retina{N - 1}Layers_width.npy")
    np.save(layerWidthVolumePath, layerWidthVolume)

print(f"=== Enf of program of generateEnfaceImages ===")


