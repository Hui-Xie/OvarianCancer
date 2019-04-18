
import numpy as np

A = np.array([[0, 1 , 2 , 3], [0, 0, 3, 3]])

print("original A = ", A)

with np.nditer(A, op_flags=['readwrite']) as it:
    for x in it:
        if 3 == x:
            x[...] = 0

print("modified A = ", A)


maxLabel = 3

remainedLabels = (0,2)

totalLabels = [x for x in range(maxLabel+1)]
for i, x in enumerate(totalLabels):
    if x in remainedLabels:
       del totalLabels[i]

print(totalLabels)


