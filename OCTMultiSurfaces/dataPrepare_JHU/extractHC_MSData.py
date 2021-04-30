# extract HC and MS training data from full training set:
nHC = 1
nMS = 2
srcDir = "/home/hxie1/data/OCT_JHU/numpy/training"
dstDir = f"/home/hxie1/data/OCT_JHU/numpy_{nHC:1d}HC_{nMS:1d}MS/training"
B = 49  # 49 bscans per patient
H = 128
W = 1024
N = 9 # number of surfaces

'''
In training data:
total 735 images = 49 Bscans/patient x 15 patients.
in 15 patients in training data:
   A first 6 patients are HC;
   B the second 9 patients are MS.
if we consider to get 1 HC patient, 2 MS patient, total 3/15 = 20% of original training data.
if we consider to get 1 HC patient, 1 MS patient, total 2/15 = 13.3% of original training data.
'''

import numpy as np
import os
import json

imagesPath = os.path.join(srcDir, "images.npy")
surfacesPath = os.path.join(srcDir,"surfaces.npy" )
patientIDPath = os.path.join(srcDir, "patientID.json")
images = np.load(imagesPath) # size: (49x15)x128x1024
surfaces = np.load(surfacesPath)         # size: (49x15)x9x1024
with open(patientIDPath) as f:
    patientIDs = json.load(f)  # size:(49x15)x1

dstImages = np.zeros(((nHC+nMS)*B, H,W),dtype=np.float32)
dstSurfaces = np.zeros(((nHC+nMS)*B, N,W),dtype=np.float32)
dstPatientID = dict()

# get dst data
dstImages[0:B*nHC,:,:] = images[0:B*nHC,:,:]
dstImages[B*nHC:B*(nHC+nMS),:,:] = images[B*6:B*(6+nMS),:,:] 

dstSurfaces[0:B*nHC,:,:] = surfaces[0:B*nHC,:,:]
dstSurfaces[B*nHC:B*(nHC+nMS),:,:] = surfaces[B*6:B*(6+nMS),:,:]

for i in range(B*nHC):
    dstPatientID[str(i)] = patientIDs[str(i)]
for i in range(B*6,B*(6+nMS)):
    dstPatientID[str(i-B*6+B*nHC)] = patientIDs[str(i)]


# save to disk:
if not os.path.exists(dstDir):
    os.makedirs(dstDir)  # recursive dir creation

dstImagesPath = os.path.join(dstDir, "images.npy")
dstSurfacesPath = os.path.join(dstDir,"surfaces.npy" )
dstPatientIDPath = os.path.join(dstDir, "patientID.json")
np.save(dstImagesPath, dstImages)
np.save(dstSurfacesPath,  dstSurfaces)
with open(dstPatientIDPath, 'w') as fp:
    json.dump(dstPatientID, fp)

print(f"======finished training data extract to new directory: {dstDir}")
