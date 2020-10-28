# generate en-face images from segmentation result and OCT volume

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

OCTVolumeDir = "/home/hxie1/data/BES_3K/numpy_10SurfaceSeg/W512/testVolume"  # in npy, a file per volume
segXmlDir ="/home/hxie1/data/BES_3K/numpy_10SurfaceSeg/W512/10SurfPredictionXml"
outputDir ="/home/hxie1/data/BES_3K/numpy_10SurfaceSeg/W512/enfaceW512"


volumesList = glob.glob(OCTVolumeDir + f"/*_Volume.npy")
for volumePath in volumesList:
    # read volume
    volume = np.load(volumePath)  # BxHxW
    B, H, W = volume.shape

    basename = os.path.basename(volumePath)
    volumename, ext = os.path.splitext(basename)
    xmlSegName = volumename+ "_Sequence_Surfaces_Prediction.xml"
    xmlSegPath = os.path.join(segXmlDir, xmlSegName)
    # read xml segmentation into array
    volumeSeg = getSurfacesArray(xmlSegPath) # BxNxW
    B1, N, W1 = volumeSeg.shape
    assert (B == B1) and (H == W1)

    # define output emtpy array
    enfaceVolume = np.empty((N-1, B, W), dtype=np.float)
    # fill the output array
    for i in range(N-1):
        surface0 = volumeSeg[:,i,:]  # BxW
        surface1 = volumeSeg[:,i+1,:]  # BxW
        layerWith = surface1 - surface0  # BxW
        for b in range(B):
            for w in range(W):
                enfaceVolume[i, b, w] = volume[b,surface0[b,w]: surface1[b,w], w].sum()/layerWith[b,w]  #BxW

    # output file
    enFaceVolumePath = os.path.join(outputDir, f"{basename}_retina{N-1}Layers_enface.npy")
    np.save(enFaceVolumePath, enfaceVolume)


    break


