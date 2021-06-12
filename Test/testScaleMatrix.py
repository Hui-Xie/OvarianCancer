B = 3
W1 =512
W2 = 200

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