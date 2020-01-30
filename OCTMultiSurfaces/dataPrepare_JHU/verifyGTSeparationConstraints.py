# Verify GT separation constraints
# and correct wrong labels.

import glob
import json
import numpy as np

segDir = "/home/hxie1/data/OCT_JHU/preprocessedData/label"

segList = glob.glob(segDir + f"/*_spectralis_macula_v1_s1_R_*.txt")

print(f"total {len(segList)} seg files")

nCountIncorrectFiles = 0

for segFile in segList:
    with open(segFile) as json_file:
        surfaces = json.load(json_file)['bds']
    surfaces = np.asarray(surfaces)
    S,W = surfaces.shape
    surfaces0 = surfaces[0:S-1, :]
    surfaces1 = surfaces[1:S,   :]
    if np.any(surfaces0 > surfaces1):
        nCountIncorrectFiles += 1
        errorLocations = np.nonzero(surfaces0>surfaces1)
        print(f"error location at file: {segFile}:\n \t {errorLocations}")
        print(f"error surface value: {surfaces0[errorLocations]}")
        print(f"its next surface value: {surfaces1[errorLocations]}")
        print(f"its next next surface value: {surfaces1[errorLocations[0]+1, errorLocations[1]]}")
        print("\n")

print(f"total nCountIncorrectFiles = {nCountIncorrectFiles}")
print("=========End of program====== ")

# fix ground truth error: use small value to replace, intead swap, the big value


