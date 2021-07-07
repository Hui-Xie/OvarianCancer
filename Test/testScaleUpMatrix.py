B = 3
W1 = 200
W2 = 512

import numpy as np
import sys
sys.path.append("..")

from OCTData.OCTDataUtilities import scaleUpMatrix

M = scaleUpMatrix(B, W1, W2)

smallM = M[0,0:14,0:14]
print("smallM:")
print(smallM)

SumColumn = np.sum(M[1,:,:], axis=0)
print(f"sumColumn, it should = 1,SumColumn.shape= {SumColumn.shape} M= {SumColumn}")

SumRow = np.sum(M[1,:,:],axis=1)
print(f"sumRow, it should = {W2/W1}, SumRow.shape= {SumRow.shape} M= {SumRow}")

print("====Good=====")