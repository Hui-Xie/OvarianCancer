# grid search lambda using unconstrained optimization on validation data
# surfaceNet is without ReLU.
# read Q, mu, r,g into tensor.

sigma2Path = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/validation/validation_simga2.npy"
muPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/validation/validation_mu.npy"
gPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/validation/validation_gt.npy"
rPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/RiftSubnet/expDuke_20200902A_RiftSubnet/testResult/validation/validation_Rift.npy"

riftGTPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/RiftSubnet/expDuke_20200902A_RiftSubnet/testResult/validation/validation_RiftGts.npy"
outputDir = "/home/hxie1/data/OCT_Duke/numpy_slices/searchSoftLambda"

import numpy as np
import torch
import sys
import os

sys.path.append("..")
from network.OCTOptimization import computeErrorStdMuOverPatientDimMean

sys.path.append("../..")
from framework.NetTools import  columnHausdorffDist

device = torch.device('cuda:0')
slicesPerPatient = 51
hPixelSize = 3.24
N = 3  # surface number

rSource = "predictR"  # "GTR": use R ground truth

# Lambda search range:
lambda0_min, lambda0_max, lambda0_step = 0, 5, 1
lambda1_min, lambda1_max, lambda1_step = 0, 5, 1

lambda0List = list(np.arange(lambda0_min, lambda0_max, lambda0_step))
lambda1List = list(np.arange(lambda1_min, lambda1_max, lambda1_step))

Naxis0 = len(lambda0List)
Naxis1 = len(lambda1List)

# lambda0 at x axis, and lambda1 is at y axis
muErrorArray = np.ones((Naxis1, Naxis0),dtype=np.float32) * 1000.0
hausdorfDArray = np.ones((Naxis1, Naxis0, N),dtype=np.float32)*1000.0

filename = f"muErr_{rSource}__lmd0_{lambda0_min}_{lambda0_max}_{lambda0_step}__lmd1_{lambda1_min}_{lambda1_max}_{lambda1_step}"
outputErrorArrayFilename = os.path.join(outputDir, filename+".npy")
outputRecordFilename = os.path.join(outputDir, filename+"_log.txt")

def unconstrainedSoftSeparation(Q, A, Lambda, Mu, R):
    '''
    unconstrained soft separation optimization

    :param Q: BxWxNxN  diagonal matrix
    :param A: BxWx(N-1)xN
    :param Lambda: BxWx(N-1)x(N-1) diagonal matrix
    :param Mu: BxWxNx1
    :param R:  BxWx(N-1)x1
    :return: S: optimal surface location of size: BxWxNx1
                by unconstrained soft optimization formula
    '''

    P = Q+ 2*torch.matmul(torch.matmul(A.transpose(-1,-2), Lambda), A)
    PInv = P.inverse()
    Y = torch.matmul(Q,Mu) + 2*torch.matmul(torch.matmul(A.transpose(-1,-2), Lambda), R)
    S = torch.matmul(PInv, Y)
    return S

def main():

    # get Sigma2, Mu, R, r, G
    sigma2 = np.load(sigma2Path) # size: 3009x3x361
    mu = np.load(muPath)
    g  = np.load(gPath)
    r  = np.load(rPath)    # size:3009x2x361
    riftGT = np.load(riftGTPath)

    B,N1,W = sigma2.shape
    assert N == N1
    M = N - 1  # number of constraints
    q = 1.0/sigma2   # B, N, W
    '''
    Q/2 min = 0.000174444867298007
    Q/2 mean = 0.48962265253067017 
    Q/2 max = 3.4920814037323
    '''


    # switch axes of N,W
    Q = torch.from_numpy(q).to(device)  # B,N,W
    Q = Q.transpose(-1, -2)  # B,W,N
    Q = torch.diag_embed(Q)  # B,W,N,N

    Mu = torch.from_numpy(mu).to(device) # B,N,W
    Mu = Mu.transpose(-1,-2)  # B,W,N
    Mu = Mu.unsqueeze(dim=-1) # B,W,N,1

    if rSource == "predictR":
        R = torch.from_numpy(r).to(device) # B,M,W
    elif rSource == "GTR":
        R = torch.from_numpy(riftGT).to(device)  # B,M,W
    else:
        print(f"useR parameter error")
        assert False
    R = R.transpose(-1, -2)  # B,W,M
    R = R.unsqueeze(dim=-1)  # B,W,M,1

    G = torch.from_numpy(g).to(device) # B,N,W

    # generate A and Lambda
    A = (torch.eye(N, N, device=device) * -1 + torch.diag(torch.ones(M, device=device) , 1))[0:-1]  # for s_i - s_{i-1}   here d indicate device.
    A = A.unsqueeze(dim=0).unsqueeze(dim=0)
    A = A.expand(B, W, M, N)

    for i in range(Naxis0):
        for j in range(Naxis1):
            lambdaVec = torch.tensor([lambda0List[i], lambda1List[j]], device=device, dtype=torch.float32)

            Lambda = torch.diag(lambdaVec)
            Lambda = Lambda.unsqueeze(dim=0).unsqueeze(dim=0)
            Lambda = Lambda.expand(B, W, M, M)

            # optimization
            S = unconstrainedSoftSeparation(Q, A, Lambda, Mu, R) # B,W,N,1

            # compute loss
            S = S.squeeze(dim=-1)
            S = S.transpose(-1,-2) # B,N,W

            # ReLU to guarantee hard separation constraint.
            for s in range(1, N):
                S[:, s, :] = torch.where(S[:, s, :] < S[:, s - 1, :], S[:, s - 1, :], S[:, s, :])

            stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(S, G,
                                                                                                     slicesPerPatient=slicesPerPatient,
                                                                                                     hPixelSize=hPixelSize,
                                                                                                     goodBScansInGtOrder=None)

            muErrorArray[j,i] = muError

            hausdorfD = columnHausdorffDist(mu, g).reshape((1, N))
            hausdorfDArray[j,i,:] = hausdorfD

    # save output
    np.save(outputErrorArrayFilename, muErrorArray)
    with open(outputRecordFilename, "w") as file:
        file.write("======= Search Lambda in Soft Separation =========\n")
        file.write(f"sigma2Path = {sigma2Path}\n")
        file.write(f"muPath = {muPath}\n")
        file.write(f"rPath = {rPath}\n")
        file.write(f"gPath = {gPath}\n")
        file.write(f"riftGTPath = {riftGTPath}\n")
        file.write(f"the coefficient of unary terms(Q = 1.0/(sigma^2)):\n")
        file.write(f"Q/2 min = {np.amin(q) / 2}\n")
        file.write(f"Q/2 mean = {np.mean(q) / 2}\n")
        file.write(f"Q/2 max = {np.amax(q) / 2}\n")
        file.write(f"at lambda0_min= {lambda0_min}, and lambda1_min = {lambda1_min}, muError = {muErrorArray[0,0]}\n")
        file.write("===========================\n\n")

        columnHausdorffDNoReLUNoOpt = columnHausdorffDist(mu, g).reshape((1, N))
        file.write(f"HausdorffDistance in pixel for NoReLU and No optimization = {columnHausdorffDNoReLUNoOpt}")
        file.write(f"HausdorffDistance in physical size (micrometer) for NoReLU and No optimization = {columnHausdorffDNoReLUNoOpt * hPixelSize}")
        stdSurfaceError, muSurfaceError, stdError, muErrorNoReLUNoOpt = computeErrorStdMuOverPatientDimMean(mu, g,
                                                                                                            slicesPerPatient=slicesPerPatient,
                                                                                                            hPixelSize=hPixelSize,
                                                                                                            goodBScansInGtOrder=None)
        file.write(f"muError at NoReLU and No optimiztion (lambda=0) = {muErrorNoReLUNoOpt}")
        file.write("===========================\n")
        file.write(f"rSource = {rSource}\n")
        file.write(f"axis x: lambda0_min, lambda0_max, lambda0_step = {lambda0_min}, {lambda0_max}, {lambda0_step}\n")
        file.write(f"axis y: lambda1_min, lambda1_max, lambda1_step = {lambda1_min}, {lambda1_max}, {lambda1_step}\n")
        j,i = np.unravel_index(np.argmin(muErrorArray), muErrorArray.shape)
        file.write(f"min location mu error = {muErrorArray[j,i]}\n")
        file.write(f"at min mu error,its Hausdorf distance(pixel) = {hausdorfDArray[j,i,:]}\n")
        file.write(f"axis x: location: x= {i}, lambda0 = {lambda0List[i]}\n")
        file.write(f"axis y: location: y= {j}, lambda1 = {lambda1List[j]}\n")

    print(f"output dir = {outputDir}")
    print(f"=== End of search search Lambda====")

if __name__ == "__main__":
    main()
