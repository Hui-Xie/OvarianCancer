
#  test BoundaryLoss
import numpy as np
import torch
from CustomizedLoss import BoundaryLoss

boundaryLoss = BoundaryLoss(lambdaCoeff=1, k=2)

inputx = np.random.rand(3,7,8)
inputx = torch.from_numpy(inputx).to("cuda")
target = np.zeros((3,7,8), dtype=int)
for i in range(3):
    target[i, 2:6, 3:7] =1
target = torch.from_numpy(target).to("cuda")

ret = boundaryLoss(inputx, target)
