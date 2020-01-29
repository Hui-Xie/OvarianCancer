# Verify GT separation constraints

import glob
import json

segDir = "/home/hxie1/data/OCT_JHU/preprocessedData/label"

segList = glob.glob(segDir + f"/*_spectralis_macula_v1_s1_R_*.txt")

for segFile in segList:
    with open(segFile) as json_file:
        surfaces = json.load(json_file)['bds']
    surfaces = np.asarray(surfaces)
