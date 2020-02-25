# measure the output performance of OCTExplorer with ground truth corrected by doctors

explorerResultDir = "/home/hxie1/data/OCT_Tongren/OCTExplorerOutput/Control"  # extract only "*__Volume_Sequence_Surfaces_Iowa.xml" file, 50 files
# gtDir = "/home/hxie1/data/OCT_Tongren/refinedGT_20200204"  # corrected result by Tongren doctors, 47 files

# it is better to use generated numpy 10-Fold data as ground truth. Just choosing one fold is ok.
numpyGTDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet" # include training, validation, and test 3 dirs
# choosing 1 fold is ok
# surfaces_CV0.npy, patientID_CV0.json

outputDir = "/home/hxie1/data/OCT_Tongren/OCTExplorerOutput"

import os
import torch
import numpy as np
import json

import sys
sys.path.append(".")
from TongrenFileUtilities import *
sys.path.append("../network")
from OCTOptimization import computeErrorStdMuOverPatientDimMean

device = torch.device('cuda:3')   #GPU ID
slicesPerPatient = 31
hPixelSize= 3.870

# collect training, validation and test GT data and its IDs
workNumpyGTDir = os.path.join(numpyGTDir, "training")
surfacesFile = os.path.join(workNumpyGTDir, "surfaces_CV0.npy")
patientIDFile = os.path.join(workNumpyGTDir, "patientID_CV0.json")
trainingSurfaces = torch.from_numpy(np.load(surfacesFile).astype(np.float32)).to(device, dtype=torch.float)  # slice, H, W
with open(patientIDFile) as f:
    trainingPatientIDs = list(json.load(f).values())

workNumpyGTDir = os.path.join(numpyGTDir, "validation")
surfacesFile = os.path.join(workNumpyGTDir, "surfaces_CV0.npy")
patientIDFile = os.path.join(workNumpyGTDir, "patientID_CV0.json")
validationSurfaces = torch.from_numpy(np.load(surfacesFile).astype(np.float32)).to(device, dtype=torch.float)  # slice, H, W
with open(patientIDFile) as f:
    validationPatientIDs = list(json.load(f).values())

workNumpyGTDir = os.path.join(numpyGTDir, "test")
surfacesFile = os.path.join(workNumpyGTDir, "surfaces_CV0.npy")
patientIDFile = os.path.join(workNumpyGTDir, "patientID_CV0.json")
testSurfaces = torch.from_numpy(np.load(surfacesFile).astype(np.float32)).to(device, dtype=torch.float)  # slice, H, W
with open(patientIDFile) as f:
    testPatientIDs = list(json.load(f).values())

gtSurfaces = torch.cat((trainingSurfaces, validationSurfaces, testSurfaces), dim=0)  # total: 47*31 = 1457 slice, size: 1457*9*512
S,N,W = gtSurfaces.shape
gtPatientIDs = trainingPatientIDs + validationPatientIDs + testPatientIDs            # total: 1457 elements of a list
assert S == len(gtPatientIDs)
# element example: '/home/hxie1/data/OCT_Tongren/control/2700_OD_6256_Volume/20110516043247_OCT01.jpg'

# according ID, collect the output of OCTExplorer
explorerSurfaces = torch.zeros_like(gtSurfaces)
patientIDList = []
s = 0
while s<S:
    patientSlicePath = gtPatientIDs[s]
    assert "_OCT01.jpg" in patientSlicePath  # make sure gtPatientIDs has sorted order
    patientID = extractPaitentID(patientSlicePath)
    patientIDList.append(patientID)

    patientSegFile = os.path.join(explorerResultDir, patientID+"_Volume_Sequence_Surfaces_Iowa.xml")
    patientSurfacesArray = getSurfacesArray(patientSegFile)

    Z, Num_Surfaces, X = patientSurfacesArray.shape
    assert 11==Num_Surfaces and Z==slicesPerPatient
    if 11 == Num_Surfaces:
        patientSurfacesArray = np.delete(patientSurfacesArray, 8, axis=1)  # delete inaccurate surface 8
        B, Num_Surfaces, X = patientSurfacesArray.shape
        assert B == 31 and Num_Surfaces == 10 and X == 768

        patientSurfacesArray = np.delete(patientSurfacesArray, 2, axis=1)  # delete inaccurate surface 2
        B, Num_Surfaces, X = patientSurfacesArray.shape
        assert B == 31 and Num_Surfaces == 9 and X == 768

    # remove the leftmost and rightmost 128 columns for each B-scans as the segmentation is not accurate
    if "5363_OD_25453" == patientID:
        patientSurfacesArray = patientSurfacesArray[:, :, 103:615]  # left shift 25 pixels for case 5363_OD_25453
    else:
        patientSurfacesArray = patientSurfacesArray[:, :, 128:640]
    explorerSurfaces[s:s + Z, :, :] = torch.from_numpy(patientSurfacesArray).to(device, dtype=torch.float)

    s += slicesPerPatient

# comparison, and report result
stdSurfaceError, muSurfaceError, stdError, muError  = computeErrorStdMuOverPatientDimMean(explorerSurfaces, gtSurfaces,
                                                                                  slicesPerPatient=slicesPerPatient,
                                                                                  hPixelSize=hPixelSize)
# check violating constraint cases:
predict0 = explorerSurfaces[:, 0:-1, :]
predict1 = explorerSurfaces[:, 1:, :]
violateConstraintErrors = torch.nonzero(predict0 > predict1, as_tuple=True)  # return as tuple

# final output result:
with open(os.path.join(outputDir, "ControlMeasureWithGT.txt"), "w") as file:
    file.write(f"Expriment Name: Measure Error between the output of OCTExplorer and corrected GT by doctor for Control group\n")
    file.write(f"explorerResultDir: {explorerResultDir}\n")
    file.write(f"ground truth Dir: {numpyGTDir}\n")
    file.write(f"S,N,W = {S, N, W}\n")
    file.write(f"\n\n===============Formal Output Result ===========\n")
    file.write(f"stdSurfaceError = {stdSurfaceError}\n")
    file.write(f"muSurfaceError = {muSurfaceError}\n")
    file.write(f"patientIDList ={patientIDList}\n")
    file.write(f"stdError = {stdError}\n")
    file.write(f"muError = {muError}\n")
    file.write(f"pixel number of violating surface-separation constraints: {len(violateConstraintErrors[0])}\n")

print("=======End of Measure OCTExplorer Output==================")
