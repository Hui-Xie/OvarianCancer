# check all layer constraints

segDir = "/home/hxie1/data/IVUS/Training_Set/Data_set_B/LABELS"
H,W = 384,384

import glob
from numpy import genfromtxt
import numpy as np
import sys
sys.path.append(".")
from PolarCartesianConverter import PolarCartesianConverter

segLumenList = glob.glob(segDir + f"/lum_frame_*_003.txt")  # e.g. lum_frame_01_0001_003.txt
segLumenList.sort()
segMediaList = glob.glob(segDir + f"/med_frame_*_003.txt")  # e.g. med_frame_01_0030_003.txt
segMediaList.sort()

print(f"total {len(segLumenList)} seg lumen files")
print(f"total {len(segMediaList)} seg media files")

assert len(segLumenList) == len(segMediaList)

N = len(segLumenList)

nCountIncorrectFiles = 0

imageShape = (H,W)
polarConverter = PolarCartesianConverter(imageShape, W//2,H//2, min(W//2,H//2), 360)

for i in range(N):
    lumenLabel = genfromtxt(segLumenList[i], delimiter=',')
    mediaLabel = genfromtxt(segMediaList[i], delimiter=',')

    label = np.array([lumenLabel, mediaLabel])
    polarLabel = polarConverter.cartesianLabel2Polar(label, rotation=0) # output size: C*N*2, in (t,r)

    if i==0:
        pC,pN,_ = polarLabel.shape
        print (f"c=0,\tN,\t t,\t r; \t\t c=1,\tN,\t t,\t r")
        for j in range(pN):
            print(f"c=0,\t{j},\t {polarLabel[0,j,0]},\t  {polarLabel[0,j,1]}; \t\t c=1,\t{j},\t {polarLabel[1,j,0]},\t {polarLabel[1,j,1]}")
        print(f"c=0,\tN,\t t,\t r; \t\t c=1,\tN,\t t,\t r")

    surfaces0 = polarLabel[0,:, 1]
    surfaces1 = polarLabel[1,:, 1]
    if np.any(surfaces0 > surfaces1):
        nCountIncorrectFiles += 1
        errorLocations = np.nonzero(surfaces0>surfaces1)
        print(f"error location at file: {segLumenList[i]}:\n \t {errorLocations}")
        print(f"error surface value: {surfaces0[errorLocations]}")
        print(f"its next surface value: {surfaces1[errorLocations]}")
        print(f"its next next surface value: {surfaces1[errorLocations[0]+1, errorLocations[1]]}")
        print("\n")

print(f"total nCountIncorrectFiles = {nCountIncorrectFiles} in {segDir}")
print("=========End of program====== ")
