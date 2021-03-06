
H=496
W=512

sliceDir = "/home/hxie1/data/BES_3K/W512AllSlices"

import numpy as np
import glob
import os

slicesList = glob.glob(sliceDir +"/*_*_*_Slice??.npy")
slicesList.sort()

errorImageID = []
for slice in slicesList:
    image = np.load(slice)
    if (H,W) != image.shape:
        print(f"image.shape={image.shape}: {slice}")
        name, _ = os.path.splitext(os.path.basename(slice))
        ID = name[0:name.find('_')]
        if ID not in errorImageID:
            errorImageID.append(ID)
        os.remove(slice)

print(f"errorImageID = \n{errorImageID}")






