# grid search lambda using unconstrained optimization on validation data
# surfaceNet is without ReLU.
# read Q, mu, r,g into tensor.

sigma2Path = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/validation/validation_simga2.npy"
muPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/validation/validation_mu.npy"
gPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/validation/validation_gt.npy"
rPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/RiftSubnet/expDuke_20200902A_RiftSubnet/testResult/validation/validation_Rift.npy"

riftGTPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/RiftSubnet/expDuke_20200902A_RiftSubnet/testResult/validation/validation_RiftGts.npy"

initialLambda = 4.0
step = 0.001

slicesPerPatient = 51
hPixelSize = 3.24


import numpy as np
import torch
import sys

sys.path.append("..")
from network.OCTOptimization import computeErrorStdMuOverPatientDimMean

device = torch.device('cuda:1')

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

#todo: A add Lamabda search
#      B use ground truth experiment


def main():

    # get Sigma2, Mu, R, r, G
    sigma2 = np.load(sigma2Path) # size: 3009x3x361
    mu = np.load(muPath)
    g  = np.load(gPath)
    r  = np.load(rPath)    # size:3009x2x361
    riftGT = np.load(riftGTPath)

    B,N,W = sigma2.shape
    M = N - 1  # number of constraints
    q = 1.0/sigma2   # B, N, W
    print(f"Q/2 min = {np.amin(q)/2}")
    print(f"Q/2 mean = {np.mean(q)/2} ")
    print(f"Q/2 max = {np.amax(q)/2}")
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

    R = torch.from_numpy(r).to(device) # B,M,W
    R = R.transpose(-1, -2)  # B,W,M
    R = R.unsqueeze(dim=-1)  # B,W,M,1

    G = torch.from_numpy(g).to(device) # B,N,W

    # generate A and Lambda
    A = (torch.eye(N, N, device=device) * -1 + torch.diag(torch.ones(M, device=device) , 1))[0:-1]  # for s_i - s_{i-1}   here d indicate device.
    A = A.unsqueeze(dim=0).unsqueeze(dim=0)
    A = A.expand(B, W, M, N)

    lambda0 = 0.48
    Lambda = torch.diag(torch.ones(M, device=device) * lambda0)
    Lambda = Lambda.unsqueeze(dim=0).unsqueeze(dim=0)
    Lambda = Lambda.expand(B, W, M, M)

    # optimization
    S = unconstrainedSoftSeparation(Q, A, Lambda, Mu, R) # B,W,N,1

    # compute loss
    S = S.squeeze(dim=-1)
    S = S.transpose(-1,-2) # B,N,W

    stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(S, G,
                                                                                             slicesPerPatient=slicesPerPatient,
                                                                                             hPixelSize=hPixelSize,
                                                                                             goodBScansInGtOrder=None)













    print(f"=== End of search search Lambda===")





if __name__ == "__main__":
    main()
