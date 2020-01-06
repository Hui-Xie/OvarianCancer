
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

def gauranteeSurfaceOrder(mu, sortedS):
    if torch.all(mu.eq(sortedS)):
        return mu
    B,surfaceNum,W = mu.size()
    S = sortedS.clone()
    for i in range(1,surfaceNum-1): # ignore consider the top surface and bottom surface
        S[:, i, :] = torch.where(mu[:, i, :] > S[:, i, :], S[:, i + 1, :], S[:, i, :])
        S[:, i, :] = torch.where(mu[:, i, :] < S[:, i, :], S[:, i - 1, :], S[:, i, :])
    return S

def main():
    mu = (torch.rand(3,5,6)*1000).to(dtype=torch.int32)
    sortedS, _ = torch.sort(mu,dim=-2)
    print(f"mu= \n{mu}")
    print(f"sortedS = \n{sortedS}")
    S = gauranteeSurfaceOrder(mu, sortedS)
    print(f"S = \n{S}")

    print("==================")


if __name__ == "__main__":
    main()
