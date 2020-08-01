# test smooth GT

import torch
import sys
sys.path.append("../OCTMultiSurfaces/network")
from OCTAugmentation import *

rawGT = torch.Tensor([[1,2,3,4, 5],[4,5,6,7,8]])


print(rawGT)

smoothedGT = smoothCMA(rawGT, 3, "reflect")  # paddingMode: 'constant', 'reflect', 'replicate' or 'circular'.

print(smoothedGT)

print("=================")