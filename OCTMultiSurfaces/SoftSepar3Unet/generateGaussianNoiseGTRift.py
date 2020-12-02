# generate Gaussian Noise with GT

sigmaList=[0.85, 0.95, 1.05, 1.15]
riftGTPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/RiftSubnet/expDuke_20200902A_RiftSubnet/testResult/validation/validation_RiftGts.npy"
outputDir = "/home/hxie1/data/OCT_Duke/numpy_slices/searchSoftLambda"

import numpy as np
import torch
import sys
import os

sys.path.append("..")
from network.OCTOptimization import computeErrorStdMuOverPatientDimMean

device = torch.device('cuda:0')
slicesPerPatient = 51
hPixelSize = 3.24
N = 3  # surface number

riftGT = np.load(riftGTPath)
RiftGT = torch.from_numpy(riftGT).to(device)

for sigma in sigmaList:
    noise = np.random.normal(0,sigma, riftGT.shape)
    noiseRift = riftGT + noise

    stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(torch.from_numpy(noiseRift).to(device), RiftGT,
                                                                                             slicesPerPatient=slicesPerPatient,
                                                                                             hPixelSize=hPixelSize, goodBScansInGtOrder=None)
    print(f"Gaussian noise sigma = {sigma} for below rift(thickness) error:")
    print(f"\tstdRiftError = {stdSurfaceError}")
    print(f"\tRiftError = {muSurfaceError}")
    print(f"\tstdRiftError = {stdError}")
    print(f"\triftError = {muError}")
    print("========================")

    basename, ext = os.path.splitext(os.path.basename(riftGTPath))
    outputFilename = basename+f"_noised_sigma_{sigma}" +ext
    outputPath = os.path.join(outputDir, outputFilename)
    np.save(outputPath, noiseRift)

print(f"=========End of generate Guassian nosed GT=============")

