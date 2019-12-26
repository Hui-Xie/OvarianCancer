# truncate OCT Jpg image at both ends in x direction

import glob
import os
from imageio import imread, imwrite

originVolumes = "/home/hxie1/data/OCT_Beijing/control"
outputVolumes = "/home/hxie1/data/OCT_Beijing/truncatedControl"
H = 496
W = 768

volumesList = glob.glob(originVolumes + f"/*_Volume")
for volume in volumesList:
    outputVolumeDir = os.path.join(outputVolumes, os.path.basename(volume))
    if not os.path.isdir(outputVolumeDir):
        os.mkdir(outputVolumeDir)
    imagesList = glob.glob(volume+f"/*_OCT[0-3][0-9].jpg")
    for imagePath in imagesList:
        filename = os.path.basename(imagePath)
        data = imread(imagePath)
        assert (H,W) == data.shape
        data = data[:,128:640]
        imwrite(os.path.join(outputVolumeDir,filename), data)

print(f"Now all truncated images are in {outputVolumes}")
