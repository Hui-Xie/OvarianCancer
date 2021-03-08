# grid search lambda using unconstrained optimization on validation data on real data:
'''
surfaces: /home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20200902A_SurfaceSubnet/testResult/test
thickness: /home/hxie1/data/OCT_Duke/numpy_slices/log/ThicknessSubnet_Z4/expDuke_20210204D_Thickness_YufanHe_FullHeightConv_iibi007/testResult/test
'''
# read Q, mu, r,g into tensor.

sigma2Path = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20200902A_SurfaceSubnet/testResult/test/test_sigma2_3surfaces.npy"
muPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20200902A_SurfaceSubnet/testResult/test/test_result_3surfaces.npy"
gPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20200902A_SurfaceSubnet/testResult/test/test_GT_3surfaces.npy"

rPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/ThicknessSubnet_Z4/expDuke_20210204D_Thickness_YufanHe_FullHeightConv_iibi007/testResult/test/test_result_3surfaces.npy"
riftGTPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/ThicknessSubnet_Z4/expDuke_20210204D_Thickness_YufanHe_FullHeightConv_iibi007/testResult/test/test_thicknessGT_3surfaces.npy"

outputDir = "/home/hxie1/data/OCT_Duke/numpy_slices/log/searchLambda_surface0902_thickness0204D"



import numpy as np
import torch
import sys
import os

sys.path.append("..")
from network.OCTOptimization import computeErrorStdMuOverPatientDimMean

sys.path.append("../..")
from framework.NetTools import  columnHausdorffDist

device = torch.device('cuda:3')
slicesPerPatient = 51
hPixelSize = 3.24
N = 3  # surface number

rSource = "predictR"  # "GTR": use R ground truth

# todo : fill lambda value with searched lambda from validation data.
lambdaVec = torch.tensor([2.555, 3.9998], device=device, dtype=torch.float32)


filename = f"muErr_{rSource}_fixLambda_lmd0_{lambdaVec[0]:.3f}_lmd1_{lambdaVec[1]:.3f}"
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
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    # get Sigma2, Mu, R, r, G
    sigma2 = np.load(sigma2Path) # size: 3009x3x361
    mu = np.load(muPath).astype(np.float32)
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

    hausdorfD = columnHausdorffDist(S.cpu().numpy(), g).reshape((1, N))


    # save output
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
        file.write("===========================\n\n")
        columnHausdorffDNoReLUNoOpt = columnHausdorffDist(mu, g).reshape((1, N))
        file.write(f"HausdorffDistance in pixel of surface branch = {columnHausdorffDNoReLUNoOpt}\n")
        file.write(f"HausdorffDistance in physical size (micrometer) of surface branch = {columnHausdorffDNoReLUNoOpt * hPixelSize}\n")
        stdSurfaceError, muSurfaceError, stdError, muErrorNoReLUNoOpt = computeErrorStdMuOverPatientDimMean(torch.from_numpy(mu).to(device), torch.from_numpy(g).to(device),
                                                                                                            slicesPerPatient=slicesPerPatient,
                                                                                                            hPixelSize=hPixelSize,
                                                                                                            goodBScansInGtOrder=None)
        file.write(f"muError of surface branch (lambda=0) = {muErrorNoReLUNoOpt}\n")
        file.write("===========================\n\n")
        file.write(f"rSource = {rSource}\n")
        file.write(f"lambda= {lambdaVec}, at test set\n")
        file.write(f"mu error with optimization on 2 branches = {muError}\n")
        file.write(f"its Hausdorf distance(pixel) = {hausdorfD}\n")

    print(f"output dir = {outputDir}")
    print(f"=== End of test with a fixed lambda ====")

if __name__ == "__main__":
    main()
