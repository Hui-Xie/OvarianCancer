B = 3
W= 200

import numpy as np
import sys
sys.path.append("..")

from OCTSegTool.thinRetina.utilities import get3PointSmoothMatrix

M = get3PointSmoothMatrix(B, W)

topM = M[0,0:10,0:10]
print("normalization top matrix:")
print(topM)

bottomM = M[0,-10:, -10:]
print("normalization bottom matrix:")
print(bottomM)

MAfterNorm = np.sum(M[1,:,:], axis=0)
print(f"Afternorm, size={MAfterNorm.shape}, M= {MAfterNorm}")

print("====Good=====")