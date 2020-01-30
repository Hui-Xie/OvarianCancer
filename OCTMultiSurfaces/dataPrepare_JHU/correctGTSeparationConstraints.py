# Verify GT separation constraints and correct wrong labels, output json file

import glob
import json
import numpy as np
import os

inputSegDir = "/home/hxie1/data/OCT_JHU/preprocessedData/label"
correctedSegDir = "/home/hxie1/data/OCT_JHU/preprocessedData/correctedLabel"

segList = glob.glob(inputSegDir + f"/*_spectralis_macula_v1_s1_R_*.txt")

print(f"total {len(segList)} seg files")

nCountIncorrectFiles = 0

for segFile in segList:
    patientIDBsan = os.path.splitext(os.path.basename(segFile))[0]
    with open(segFile) as json_file:
        surfaces = json.load(json_file)['bds']
    surfaces = np.asarray(surfaces)
    S,W = surfaces.shape
    surfaces0 = surfaces[0:S-1, :]
    surfaces1 = surfaces[1:S,   :]
    if np.any(surfaces0 > surfaces1):
        nCountIncorrectFiles += 1
        surfaces = np.sort(surfaces, axis=-2)
    boundaryDict = {}
    boundaryDict['bds'] = surfaces.tolist()
    jsonData = json.dumps(boundaryDict)
    with open(os.path.join(correctedSegDir, patientIDBsan+".json"), "w") as f:
        f.write(jsonData)

print(f"total nCountIncorrectFiles = {nCountIncorrectFiles}")
print(f"corrected file output at {correctedSegDir}")
print("=========End of program====== ")

# fix ground truth error: use small value to replace, intead swap, the big value
