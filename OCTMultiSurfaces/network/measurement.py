
# some measurement function for OCT Multisurface segmentation

import torch

def computeMuVariance(x):
    '''
    Compute the mean and variance along H direction of each surface.

    :param x: in (BatchSize, NumSurface, H, W) dimension, the value is probability (after Softmax) along each Height direction
    :return: mu:     mean in (BatchSize, NumSurface, W) dimension
             sigma2: variance in (BatchSize, Numsurface, W) dimension.
    '''
    device = x.device
    B,Num,H,W = x.size() # Num is the num of surface for each patient

    # square probability to strengthen the big probability, and to reduce variance
    # "The rich get richer and the poor get poorer"
    P = torch.pow(x, 2).float()
    PColSum = torch.sum(P, dim=-2, keepdim=True).expand(P.size())  # column means H direction
    P = P/PColSum
    del PColSum   # in order to free memory for further reuse.

    # compute mu
    Y = torch.arange(H).view((H,1)).expand(P.size()).float().to(device)
    mu = torch.sum(P*Y, dim=-2, keepdim=True)

    # compute sigma2 (variance)
    Mu = mu.expand(P.size())
    sigma2 = torch.sum(P*torch.pow(Y-Mu,2), dim=-2,keepdim=False)

    return mu.squeeze(dim=-2),sigma2

def computeErrorStdMu(predicitons, gts, slicesPerPatient=31, hPixelSize=3.780):
    '''
    Compute error standard deviation and mean along diffrent dimension.


    :param predicitons: in (BatchSize, NumSurface, W) dimension, in strictly patient order.
    :param gts: in (BatchSize, NumSurface, W) dimension
    :param hPixelSize: in micrometer
    :return: muSurface: (NumSurface) dimension, mean for each surface
             stdSurface: (NumSurface) dimension
             muPatient: (NumPatient) dimension, mean for each patient
             stdPatient: (NumPatient) dimension
             mu: a scalar, mean over all surfaces and all batchSize
             std: a scalar
    '''
    N,NumSurface, W = predicitons.shape

    Error = torch.abs(predicitons-gts)
    stdSurface, muSurface = tuple(x*hPixelSize for x in torch.std_mean(Error, dim=(0,2)))
    stdPatient = torch.FloatTensor([torch.std(Error[i * slicesPerPatient:(i + 1) * slicesPerPatient, ]) for i in range(N // slicesPerPatient)]) *hPixelSize
    muPatient  = torch.FloatTensor([torch.mean(Error[i * slicesPerPatient:(i + 1) * slicesPerPatient, ]) for i in range(N // slicesPerPatient)]) *hPixelSize
    std, mu = tuple(x*hPixelSize for x in torch.std_mean(Error))
    return stdSurface, muSurface, stdPatient,muPatient, std,mu




