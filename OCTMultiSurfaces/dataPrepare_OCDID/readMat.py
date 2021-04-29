import numpy as np
from scipy.io import loadmat

matFilePath =  "/home/hxie1/data/OCTID/rawData/normalSeg/198_OS_octSegmentation.mat"

matInfo = loadmat(matFilePath)

for key in matInfo:
    if "__" in key:
        print(f"{key} = {matInfo[key]}")
    else:
        print(f"{key}.size = {matInfo[key].shape}")

print("==================")
