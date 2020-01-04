
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
x = torch.tensor([5.0, 1.0, 3.0, 2.0], requires_grad=True)
x1, _ = torch.sort(x)
y = torch.pow(x1,2.0).sum()
y.backward()

print(f"x.grad = {x.grad}")

nIterations = 150
mu = torch.tensor([3.0,2.0,1.0,5])
S0, _ = torch.sort(mu)
learningStep = 0.01
sigma2 = 1


# IPM iteration
S=S0
for i in range(nIterations):
    S1 = S0-learningStep*(S0-mu)/sigma2
    error = ((S1 - mu) ** 2 / sigma2).sum()
    S2, _ = torch.sort(S1)
    S0 = S2
    print(f"i= {i}, S1={S1}, S2={S2}. sort= {not torch.all(S2.eq(S1))}, error={error}")
    if torch.all(S2.eq(S1)):
        S = S1
        continue
    else:
        break

print( f"S = {S}")

print("==================")
