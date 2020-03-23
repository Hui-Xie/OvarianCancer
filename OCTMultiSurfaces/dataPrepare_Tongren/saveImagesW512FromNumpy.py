# save image of width of 512 from numpy, and output them
numpyDir = "/home/hxie1/data/OCT_Tongren/numpy/glaucomaRaw_W512"
outputDir = "/home/hxie1/data/OCT_Tongren/glaucomaImages_W512"

from imageio import imwrite
import numpy as np
import json
import os

import sys
sys.path.append(".")
from TongrenFileUtilities import extractFileName

images = np.load(os.path.join(numpyDir,"test",f"images.npy"))
patientIDPath = os.path.join(numpyDir, 'test', f"patientID.json")
with open(patientIDPath) as f:
    patientIDs = json.load(f)

NSlice,H,W = images.shape
assert NSlice==49*31 and H==496 and W==512

i=0
while i<NSlice:
    dirPath, fileName = os.path.split(patientIDPath[str(i)])
    dirName = os.path.basename(dirPath)
    newDirPath = os.path.join(outputDir,dirName)
    if not os.path.exists(newDirPath):
        os.makedirs(newDirPath)  # recursive dir creation
    for j in range(i, i+31):
        image = images[j,:,:]
        imwrite(os.path.join(newDirPath, fileName), image)
    i +=31

print("====End of output all image of Width 512==============")

