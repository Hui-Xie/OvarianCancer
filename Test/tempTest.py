
# keep this:
'''

m = 2
nCount = 0
for a in range(-m, m+1):
    for b in range(-m, a+1):
        for c in range(-m, m+1):
            if (a==b and a>0) or (a==b and a==0 and c>=0):
                continue

            nCount +=1
            print(f"(a,b, c)={a,b,c}")

print(f"nCount = {nCount}")

'''

'''
import numpy as np

numpyFile1 = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/training/latent/latent_20191207_135106/05253160.npy"
numpyFile2 = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/training/latent/latent_20191207_135106/05311044.npy"
V1 = np.load(numpyFile1)
V2 = np.load(numpyFile2)

VDiff = V1-V2

Vstd = np.std(VDiff)

print(f"Vstd = {Vstd}")
'''



import torch
#x = torch.tensor([1.0+2.0*1j], requires_grad=True, dtype=torch.complex32)
x = torch.tensor([-1.0], requires_grad=True)
y = torch.pow(x,1.0/3)+3.0
y.backward()

print(f"x.grad = {x.grad}")



print("==================")
