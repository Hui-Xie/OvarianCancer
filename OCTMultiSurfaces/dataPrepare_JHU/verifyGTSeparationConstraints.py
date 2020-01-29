# Verify GT separation constraints

import glob
import json
import numpy as np

segDir = "/home/hxie1/data/OCT_JHU/preprocessedData/label"

segList = glob.glob(segDir + f"/*_spectralis_macula_v1_s1_R_*.txt")

print(f"total {len(segList)} seg files")

for segFile in segList:
    with open(segFile) as json_file:
        surfaces = json.load(json_file)['bds']
    surfaces = np.asarray(surfaces)
    S,W = surfaces.shape
    surfaces0 = surfaces[0:S-1, :]
    surfaces1 = surfaces[1:S,   :]
    if np.any(surfaces0 > surfaces1):
        errorLocations = np.nonzero(surfaces0>surfaces1)
        print(f"error location at file: {segFile}:\n \t {errorLocations}")
        print(f"surface value: {surfaces0[errorLocations]}")
        print(f"its next layer value: {surfaces1[errorLocations]}")
        print("\n")
print("=========End of program====== ")


