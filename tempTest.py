
import numpy as np

A = np.array([[0, 1 , 2 , 3], [0, 0, 3, 3]])


maxLabel = 3

remainedLabels = (0,2)

totalLabels = [x for x in range(maxLabel+1)]
for i, x in enumerate(totalLabels):
    if x in remainedLabels:
       del totalLabels[i]

print(totalLabels)


