
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
    '''



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
    mu = torch.tensor([1.0,5.0,3,2,6,7,9,12])
    N = len(mu)
    LIS = getLIS(mu)
    print(f"mu = {mu}")
    print(f"LIS = {LIS}")
    mu = mu.view(1,N,1)
    LIS = LIS.view(1,N,1)
    confusionLIS = markConfusionSectionFromLIS(mu, LIS)
    print(f"confusionLIS ={confusionLIS.view(N)}")
    sigma2 = torch.empty_like(mu).fill_(1.0)
    S = constraintOptimization(mu, sigma2, confusionLIS)
    print(f"final S ={S.view(N)}")

    print(f"\ntest disroder region")
    mu = torch.tensor([1.0, 5, 3, 2, 6, 9, 8, 10, 12])
    N = len(mu)
    LIS = getLIS(mu)
    print(f"mu = {mu}")
    print(f"LIS = {LIS}")
    mu = mu.view(1, N, 1)
    LIS = LIS.view(1, N, 1)
    confusionLIS = markConfusionSectionFromLIS(mu, LIS)
    print(f"confusionLIS ={confusionLIS.view(N)}")
    sigma2 = torch.empty_like(mu).fill_(1.0)
    S = constraintOptimization(mu, sigma2, confusionLIS)
    print(f"final S ={S.view(N)}")

    print(f"\ntest disroder region")
    mu = torch.tensor([4.0, 5, 3, 2, 6, 9, 8, 10, 12])
    N = len(mu)
    LIS = getLIS(mu)
    print(f"mu = {mu}")
    print(f"LIS = {LIS}")
    mu = mu.view(1, N, 1)
    LIS = LIS.view(1, N, 1)
    confusionLIS = markConfusionSectionFromLIS(mu, LIS)
    print(f"confusionLIS ={confusionLIS.view(N)}")
    sigma2 = torch.empty_like(mu).fill_(1.0)
    S = constraintOptimization(mu, sigma2, confusionLIS)
    print(f"final S ={S.view(N)}")

    print(f"\ntest disroder region")
    mu = torch.tensor([5, 2.0, 3, 5, 6, 9, 8, 10, 12])
    N = len(mu)
    LIS = getLIS(mu)
    print(f"mu = {mu}")
    print(f"LIS = {LIS}")
    mu = mu.view(1, N, 1)
    LIS = LIS.view(1, N, 1)
    confusionLIS = markConfusionSectionFromLIS(mu, LIS)
    print(f"confusionLIS ={confusionLIS.view(N)}")
    sigma2 = torch.empty_like(mu).fill_(1.0)
    S = constraintOptimization(mu, sigma2, confusionLIS)
    print(f"final S ={S.view(N)}")

    print("==================")

    print ("test gauranteeSurfaceOrder")
    S = torch.tensor([1, 5, 3, 2, 6, 7, 8, 9])
    N = len(S)
    LIS = getLIS(S)
    S = S.view(1,N,1)
    print(f"S input = {S.view(N)}")

    LIS = LIS.view(1, N, 1)
    S = gauranteeSurfaceOrder(S, LIS)
    print(f"S output = {S.view(N)}")
    print("")

    S = torch.tensor([1, 3, 2, 5, 6, 7, 8, 9])
    N = len(S)
    LIS = getLIS(S)
    S = S.view(1, N, 1)
    print(f"S input = {S.view(N)}")

    LIS = LIS.view(1, N, 1)
    S = gauranteeSurfaceOrder(S, LIS)
    print(f"S output = {S.view(N)}")
    print("")

    S = torch.tensor([1, 4, 5, 2, 6, 7, 8, 9])
    N = len(S)
    LIS = getLIS(S)
    S = S.view(1, N, 1)
    print(f"S input = {S.view(N)}")
    # S = fillGapOfLIS(getBatchLIS(S), S)

    LIS = LIS.view(1, N, 1)
    S = gauranteeSurfaceOrder(S, LIS)
    print(f"S output = {S.view(N)}")
    print("")

    '''

    '''
    print(f"======================test parallel getLIS in GPU======================")
    X = (torch.rand(2,11,5)*100).int()
    '''

    '''
    X = torch.tensor([2,8,5,2,1])
    print(f"X = {X}")
    x1D =getLIS(X)
    print(f"X1D LIS = {x1D}")
    X = X.view(1,torch.numel(X),1)
    '''

    '''
    # X.to(torch.device('cuda:0'))
    print(f"X = \n{X}\n")
    LIS_X = getBatchLIS_gpu(X)
    print(f"LIS_X =\n{LIS_X}")
    '''

    print("Test dynamic programmming")
    B, NumSurfaces, H, W = 3,5,8,4
    logP = torch.rand(B, NumSurfaces, H, W)

    '''
    logP = torch.tensor(
           [[0.1, 0.5, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.4, 0.1, 0.2, 0.1, 0.2, 0.1],
            [0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.4],
            [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.8],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1]])
    '''


    logP = logP.view(B, NumSurfaces, H, W)
    print(f"logP: size= {logP.shape}")
    print(f"logP = \n{logP.squeeze()}")
    L = DPComputeSurfaces(logP)
    print(f"L = {L}")






if __name__ == "__main__":
    main()
