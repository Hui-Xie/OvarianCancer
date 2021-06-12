B = 3
W1 =512
W2 = 200

import numpy as np
import sys
sys.path.append("..")

from OCTSegTool.thinRetina.utilities import scaleMatrix

M = scaleMatrix(B, W1,W2)

smallM = M[0,0:10,0:10]
print("normalization matrix:")
print(smallM)

smallM = M[1,0:14,0:14]
print("before Normalization:")
s = W1/W2
print(f"s= {s}")
print(smallM*s)

MAfterNorm = np.sum(M[1,:,:], axis=0)
print(f"Afternorm, size={MAfterNorm.shape}, M= {MAfterNorm}")

MBeforeNorm = np.sum(M[1,:,:]*s,axis=1)
print(f"Before Norm,size={MBeforeNorm.shape},  M= {MBeforeNorm}")

print("====Good=====")