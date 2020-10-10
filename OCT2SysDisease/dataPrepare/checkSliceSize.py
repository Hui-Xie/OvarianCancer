
H=496
W=512

sliceDir = "/home/hxie1/data/BES_3K/W512AllSlices"

import numpy as np
import glob

slicesList = glob.glob(sliceDir +"/*_*_*_Slice??.npy")
for slice in slicesList:
    image = np.load(slice)
    if (H,W) != image.shape:
        print(f"image.shape={image.shape}: {slice}")
        






