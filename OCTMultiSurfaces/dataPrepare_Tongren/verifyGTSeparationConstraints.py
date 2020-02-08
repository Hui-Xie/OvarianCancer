# Verify GT separation constraints

import glob
import json
import numpy as np
import os

segDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/test"


segFileList = glob.glob(os.path.join(segDir,"surfaces_CV*.npy"))

print(f"total {len(segFileList)} surfaces files.")

nCountIncorrectFiles = 0

for segFile in segFileList:

    surfaces = np.load(segFile)
    S,N, W = surfaces.shape
    surfaces0 = surfaces[:, 0:N-1, :]
    surfaces1 = surfaces[:, 1:N,   :]
    if np.any(surfaces0 > surfaces1):
        nCountIncorrectFiles += 1
        errorLocations = np.nonzero(surfaces0>surfaces1)
        print(f"error location at file: {segFile}:\n \t {errorLocations}")
        print(f"error surface value: {surfaces0[errorLocations]}")
        print(f"its next surface value: {surfaces1[errorLocations]}")
        print(f"its next next surface value: {surfaces1[errorLocations[0]+1, errorLocations[1]]}")
        print("\n")

print(f"total nCountIncorrectFiles = {nCountIncorrectFiles} in {segDir}")
print("=========End of program====== ")

# fix ground truth error: use small value to replace, intead swap, the big value


