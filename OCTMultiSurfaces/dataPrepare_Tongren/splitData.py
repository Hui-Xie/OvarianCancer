# from one fold CV data, split the last one patient data out for validation
imageFilePath = "/home/hxie1/data/OCT_Tongren/numpy/test/originalCV5/images_CV5.npy"
surfaceFilePath = "/home/hxie1/data/OCT_Tongren/numpy/test/originalCV5/surfaces_CV5.npy"
patientIDFilePath = "/home/hxie1/data/OCT_Tongren/numpy/test/originalCV5/patientID_CV5.json"

output1Path = "/home/hxie1/data/OCT_Tongren/numpy/test"
output2Path = "/home/hxie1/data/OCT_Tongren/numpy/validation"

nCut = 31  #we cut last nCut slides into output2Path

import numpy as np
import os
import json

# split images
images = np.load(imageFilePath)
S,W,H = images.shape
image1 = images[0:S-nCut,]
image2 = images[S-nCut:, ]
np.save(os.path.join(output1Path,"images_CV5.npy"), image1)
np.save(os.path.join(output2Path,"images_CV5.npy"), image2)


# split surfaces
surfaces = np.load(surfaceFilePath)
S,W,N = surfaces.shape
surface1 = surfaces[0:S-nCut,]
surface2 = surfaces[S-nCut:, ]
np.save(os.path.join(output1Path,"surfaces_CV5.npy"), surface1)
np.save(os.path.join(output2Path,"surfaces_CV5.npy"), surface2)

# split patientIDs
with open(patientIDFilePath) as f:
    patientIDs = json.load(f)
S = len(patientIDs)
s =0
s2=0
patientID1 ={}
patientID2 ={}
for (key, value) in patientIDs.items():
    if s<S-nCut:
        patientID1[key] = value
    else:
        patientID2[str(s2)] = value
        s2 += 1
    s +=1

with open(os.path.join(output1Path,"patientID_CV5.json"), 'w') as f:
    json.dump(patientID1, f)
with open(os.path.join(output2Path, "patientID_CV5.json"), 'w') as f:
    json.dump(patientID2, f)
