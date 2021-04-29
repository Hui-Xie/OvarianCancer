
import sys
import numpy as np


from scipy.io import loadmat

matFilePath =  "/home/hxie1/data/OCT_DME/rawZip/2015_BOE_Chiu/Subject_05.mat"

matInfo = loadmat(matFilePath)

for key in matInfo:
    if "__" in key:
        print(f"{key} = {matInfo[key]}")
    else:
        print(f"{key}.size = {matInfo[key].shape}")

# ground truth contains nan.

#keyname ="automaticLayersDME"
keyname = 'manualLayers1'  # original paper uses MJA's segmentation result.
print(f"\n\nground truth name={keyname}")
gt = matInfo[keyname]
print( "valid GT index:")
print(f"gt.size = {gt.shape}")
colStart = 128 # 116 surface location exists nan in intermediate locations.
colEnd = 652   # 658
print(f"colStart={colStart}, colEnd={colEnd}")
# its fovea is not fixed at 30.
validBscanIndex=(10,15,20,25,28,30,32,35,40,45,50)

for i in validBscanIndex:
    print(f"Bscan {i}",end=":")
    for n in range(8):  # surface
        asum = gt[n,colStart:colEnd, i].sum()
        if not np.isnan(asum):
            print(f"{n}", end=" ")
    print("")





