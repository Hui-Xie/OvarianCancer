
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
import sys
sys.path.append("../OCTMultiSurfaces/network")
from OCTOptimization import *


def main():
    A = torch.tensor([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15])
    ALIS = getLIS(A)
    print(f"A =  {A}, \nALIS= {ALIS}")

    B = torch.tensor([10, 22, 9, 33, 21, 50, 41, 60, 80])
    BLIS = getLIS(B)
    print(f"B =  {B}, \nBLIS= {BLIS}")

    B = torch.tensor([50, 3, 10, 7, 40, 80])
    BLIS = getLIS(B)
    print(f"B =  {B}, \nBLIS= {BLIS}")


    print(f"\ntest disroder region")
    mu = torch.tensor([1,5,3,2,6,7,9,12])
    N = len(mu)
    LIS = getLIS(mu)
    print(f"mu = {mu}")
    print(f"LIS = {LIS}")
    mu = mu.view(1,N,1)
    LIS = LIS.view(1,N,1)
    disorderLIS = markConfusionSectionFromLIS(mu, LIS)
    print(f"disorderLIS ={disorderLIS.view(N)}")

    print(f"\ntest disroder region")
    mu = torch.tensor([1, 5, 3, 2, 6, 9, 8, 10, 12])
    N = len(mu)
    LIS = getLIS(mu)
    print(f"mu = {mu}")
    print(f"LIS = {LIS}")
    mu = mu.view(1, N, 1)
    LIS = LIS.view(1, N, 1)
    disorderLIS = markConfusionSectionFromLIS(mu, LIS)
    print(f"disorderLIS ={disorderLIS.view(N)}")

    print(f"\ntest disroder region")
    mu = torch.tensor([4, 5, 3, 2, 6, 9, 8, 10, 12])
    N = len(mu)
    LIS = getLIS(mu)
    print(f"mu = {mu}")
    print(f"LIS = {LIS}")
    mu = mu.view(1, N, 1)
    LIS = LIS.view(1, N, 1)
    disorderLIS = markConfusionSectionFromLIS(mu, LIS)
    print(f"disorderLIS ={disorderLIS.view(N)}")

    print(f"\ntest disroder region")
    mu = torch.tensor([5, 2, 3, 5, 6, 9, 8, 10, 12])
    N = len(mu)
    LIS = getLIS(mu)
    print(f"mu = {mu}")
    print(f"LIS = {LIS}")
    mu = mu.view(1, N, 1)
    LIS = LIS.view(1, N, 1)
    disorderLIS = markConfusionSectionFromLIS(mu, LIS)
    print(f"disorderLIS ={disorderLIS.view(N)}")

    print("==================")


if __name__ == "__main__":
    main()
