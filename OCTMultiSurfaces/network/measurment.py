
# some measurement function for OCT Multisurface segmentation

import torch

def computeMuVariance(x):
    '''
    compute the mean and variance along H direction of each surface.

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

    # compute mu
    Y = torch.arange(H).view((H,1)).expand(P.size()).float().to(device)
    mu = torch.sum(P*Y, dim=-2, keepdim=True)

    # compute sigma2 (variance)
    Mu = mu.expand(P.size())
    sigma2 = torch.sum(P*torch.pow(Y-Mu,2), dim=-2,keepdim=False)

    return mu.squeeze(dim=-2),sigma2