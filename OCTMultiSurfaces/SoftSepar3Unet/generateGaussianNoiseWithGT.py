# generate Gaussian Noise with GT

sigmaList=[0.4, 0.7, 1.0, 1.3]
gPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/validation/validation_gt.npy"
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

g = np.load(gPath)
G = torch.from_numpy(g).to(device)

for sigma in sigmaList:
    noise = np.random.normal(0,sigma, g.shape)
    noiseG = g + noise

    stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(torch.from_numpy(noiseG).to(device), G,
                                                                                             slicesPerPatient=slicesPerPatient,
                                                                                             hPixelSize=hPixelSize, goodBScansInGtOrder=None)
    print(f"Gaussian noise sigma = {sigma}")
    print(f"\tstdSurfaceError = {stdSurfaceError}")
    print(f"\tmuSurfaceError = {muSurfaceError}")
    print(f"\tstdError = {stdError}")
    print(f"\tmuError = {muError}")
    print("========================")

    basename, ext = os.path.splitext(os.path.basename(gPath))
    outputFilename = basename+f"_noised_sigma_{sigma}" +ext
    outputPath = os.path.join(outputDir, outputFilename)
    np.save(outputPath, noiseG)

print(f"=========End of generate Guassian nosed GT=============")

