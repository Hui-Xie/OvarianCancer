# test to catch matrix singular exception

import torch
import time
start_time = time.time()

#A = torch.tensor(range(1,10)).view(3,3).float()
A = torch.tensor([[1.,1.0],[1.0,1.0]])
print(f"A = \n{A}")
print(f"A.det = {A.det()}")

try:
    AInv = torch.inverse(A)
except RuntimeError as err:
    print(err)
    if "singular U" in str(err):
        AInv = torch.pinverse(A)  # pinverse costs 10 times of time of inverse.
        print("except singular matrix, use pinverse")

print(f"AInv = \n{AInv}")

C = torch.matmul(A,AInv)

print(f"C = \n{C}")

B=10
W=512
N=21
device = torch.device('cuda:0')
A = torch.rand(B,W,N,N).to(device)

startTime = time.time()
AInv = torch.inverse(A)
dualTime = time.time()-startTime
print(f"Inverse time: {dualTime} seconds")


startTime = time.time()
AInv = torch.pinverse(A)
dualTime = time.time()-startTime
print(f"Pseudo Inverse time: {dualTime} seconds")


