import sys
import numpy as np
sys.path.append("..")
from CustomizedLoss import BoundaryLoss3

import torch

inputs = torch.tensor([
                   [ [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                     [[0,0,0,0,0], [0,2,2,2,0], [0,2,2,2,0],[0,0,0,0,0],[0,0,0,0,0]]],
                   [ [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                     [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0],[1,1,0,0,0],[1,1,0,0,0]]] ], dtype=torch.float32, requires_grad=True)

targets = torch.tensor([
                   [[0,1,1,1,0], [0,1,1,1,0], [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                   [[0,0,0,1,1], [0,0,0,1,1], [0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]] ])



boundaryLoss = BoundaryLoss3()

inputs= inputs.cuda()
targets = targets.cuda()

loss = boundaryLoss(inputs, targets)

print (f"Loss = {loss.item()}")

inputs.retain_grad() # Enables .grad attribute for non-leaf Tensors.
loss.backward()

print (f"inputs gradient = {inputs.grad}")