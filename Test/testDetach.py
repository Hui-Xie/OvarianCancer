# test detach

import torch

N=2
W=3
dtype = torch.float32
device = torch.device("cuda:0")

A = torch.rand(N,W, dtype=dtype, device=device, requires_grad=True)
AClone = (A+2).clone().detach()

S = (A+AClone).sum()
S.backward()

print(f"A.gradient = {A.grad}")
print(f"AClone.gradient = {AClone.grad}")

print(f"================")