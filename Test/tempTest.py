
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
from PrimalDualIPM import *


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

    '''
    print("Test dynamic programmming")
    B, NumSurfaces, H, W = 3,5,8,4
    logP = torch.rand(B, NumSurfaces, H, W)

    '''

    '''
    logP = torch.tensor(
           [[0.1, 0.5, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.4, 0.1, 0.2, 0.1, 0.2, 0.1],
            [0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.4],
            [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1, 0.8],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.1, 0.1]])
    '''

    '''
    logP = logP.view(B, NumSurfaces, H, W)
    print(f"logP: size= {logP.shape}")
    print(f"logP = \n{logP.squeeze()}")
    L = DPComputeSurfaces(logP)
    print(f"L = {L}")
    '''

    print("test Primal dual IPM method")
    ctx = None
    B,W,N= 2,3,11

    Mu = torch.tensor([1.0,2.0,3.0,4.0,6.0,5.0,7.0,8.0,9.0,10.0,11.0])
    print(f"Mu= {Mu}")
    Mu = Mu.unsqueeze(dim=0).unsqueeze(dim=0)
    Mu = Mu.expand(B,W,N)

    Q  = torch.eye(N)
    Q = Q.unsqueeze(dim=0).unsqueeze(dim=0)
    Q = Q.expand(B, W, N, N)

    A = (torch.eye(N,N)+torch.diag(torch.ones(N-1)*-1, 1))[0:-1]
    print(f"A=\n{A}")
    A = A.unsqueeze(dim=0).unsqueeze(dim=0)
    A = A.expand(B, W, N-1,N)

    S0 = torch.tensor([1.0,2.0,3.0,4.0,4.0,7.0,7.0,8.0,9.0,10.0,11.0])
    S0 = S0.unsqueeze(dim=0).unsqueeze(dim=0)
    S0 = S0.expand(B,W,N)

    Lambda = torch.rand(B,W,N-1)
    alpha = 10+ torch.rand(B,W)
    epsilon = 0.001


    S = SeparationPrimalDualIPMFunction.forward(ctx, Mu, Q, A, S0, Lambda, alpha, epsilon)
    print(f"S.shape = {S.shape}")
    print (f"S =\n{S}")










if __name__ == "__main__":
    main()
