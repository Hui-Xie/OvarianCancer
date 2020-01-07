
# some optimizatino and measurement function for OCT Multisurface segmentation

import torch
import math

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
    P = torch.pow(x, 2).to(device=device, dtype=torch.float32)  # it must use float32. float16 will lead sigma2 overflow(NaN)
    PColSum = torch.sum(P, dim=-2, keepdim=True).expand(P.size())  # column means H direction
    P = P/PColSum
    del PColSum   # in order to free memory for further reuse.
    #with torch.cuda.device(device): # using context is to avoid extra GPU using in GPU0 for empty_cache()
    #    torch.cuda.empty_cache()

    # compute mu
    Y = torch.arange(H).view((H,1)).expand(P.size()).to(device=device, dtype=torch.int16)
    # mu = torch.sum(P*Y, dim=-2, keepdim=True)
    # use slice method to comput P*Y
    for b in range(B):
        if 0==b:
            PY = (P[b,]*Y[b,]).unsqueeze(dim=0)
        else:
            PY = torch.cat((PY, (P[b,]*Y[b,]).unsqueeze(dim=0)))
    mu = torch.sum(PY, dim=-2, keepdim=True)
    del PY  # hope to free memory.
    #with torch.cuda.device(device):
    #    torch.cuda.empty_cache()

    # compute sigma2 (variance)
    Mu = mu.expand(P.size())

    #sigma2 = torch.sum(P*torch.pow(Y-Mu,2), dim=-2,keepdim=False)
    # this slice method is to avoid using big GPU memory .
    for b in range(B):
        if 0==b:
            sigma2 = torch.sum(P[b,]*torch.pow(Y[b,]-Mu[b,],2), dim=-2,keepdim=False).unsqueeze(dim=0)
        else:
            sigma2 = torch.cat((sigma2, torch.sum(P[b,]*torch.pow(Y[b,]-Mu[b,],2), dim=-2,keepdim=False).unsqueeze(dim=0)))

    return mu.squeeze(dim=-2),sigma2

def proximalIPM(mu,sigma2, sortedS, nIterations=100, learningStep=0.01):
    '''
    use proximal IPM method to optimize the final output surface location by Unet.
    It is used in inference stage.

    :param mu: mean of size (B,S,W), where S is surface
    :param sigma2: variance of size(B,S,W)
    :param sortedS: sorted S from mu in ascending order
    :param nIterations: the iteration number  of proximal IPM method
    :return:
           S: the optimized surface locations in [B,S, W] dimension.
    '''

    if torch.all(mu.eq(sortedS)):
        return sortedS

    # get initial surface locations in ascending order, which do not need gradient
    S0 = sortedS.clone().detach()
    # IPM iteration
    S=S0
    for i in range(nIterations):
        S1 = S0-learningStep*(S0-mu)/sigma2
        S2, _ = torch.sort(S1, dim=-2)
        S0 = S2
        if torch.all(S2.eq(S1)):
            S = S1
            continue
        else:  # when one swapping occures, at least one value reaches its boundary
            break

    return S

def computeErrorStdMu(predicitons, gts, slicesPerPatient=31, hPixelSize=3.870):
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


class OCTMultiSurfaceLoss():

    def __init__(self, reduction='mean'):
        self.m_reduction = reduction

    def __call__(self, Mu, Sigma2, GTs):
        loss = 0.5*torch.pow(GTs-Mu, 2)/Sigma2
        loss = loss.sum()
        if "mean" == self.m_reduction:
            num = torch.numel(Mu)
            loss /=num
        return loss


def gauranteeSurfaceOrder(mu, sortedS, sigma2):
    if torch.all(mu.eq(sortedS)):
        return mu
    B,surfaceNum,W = mu.size()
    device = mu.device()

    # method1: use upper bound and lower bound to replace its current location value
    S = sortedS.clone()
    for i in range(1,surfaceNum-1): # ignore consider the top surface and bottom surface
        S[:, i, :] = torch.where(mu[:, i, :] > S[:, i, :], S[:, i + 1, :], S[:, i, :])
        S[:, i, :] = torch.where(mu[:, i, :] < S[:, i, :], S[:, i - 1, :], S[:, i, :])

    # method2: along the H(surface)direction, if n>1 continious locations all do not equal with ist sorted location sortedS,
    #          then the inverse-variance-weigted average as one location should be one best approximate to the all orginal n disorder locations.
    #          this is a local thinking to achieve subpixel accuracy.

    # find equal location
    S = torch.zeros_like(mu)   # 0 means the location has not been process.
    S = torch.where(mu == sortedS, mu, S)

    # convert tensor to cpu
    S_cpu = S.cpu()
    sigma2_cpu = sigma2.cpu()
    mu_cpu =mu.cpu()
    sortedS_cpu = sortedS.cpu()

    # element-wise local optimization
    for b in range(0,B):
        for w in range(0,W):
            s = 0
            while(s<surfaceNum):
                if 0 == S_cpu[b,s,w]:
                    n = 1  # n continious disorder predicted locations
                    while s+n<surfaceNum and 0 == S_cpu[b,s+n,w]:
                        n += 1
                    if 1 == n:
                        S_cpu[b,s,w] = sortedS_cpu[b,s,w]
                    else:
                        numerator = 0.0
                        denominator = 0.0
                        for k in range(0, n):
                            numerator += mu_cpu[b,s+k,w]/sigma2_cpu[b,s+k,w]
                            denominator += 1.0/sigma2_cpu[b,s+k,w]
                        x = numerator/denominator
                        for k in range(0, n):
                            S_cpu[b, s+k, w] = x
                    s = s+n
                else:
                    s += 1
    S = S_cpu.to(device)

    return S


def getLIS(inputTensor):
    '''
    get Largest Increasing Subsequence  with non-choosing element marked as 0
    https://en.wikipedia.org/wiki/Longest_increasing_subsequence

    :param inputTensor:
    :return: Tensor with choosing element in its location with non-choosing element marked as 0, same length with inputTensor
    '''
    X = inputTensor
    N = len(inputTensor)
    P = torch.zeros(N, dtype=torch.long)  #  stores the index of the predecessor of X[k] in the longest increasing subsequence ending at X[k].
    M = torch.zeros(N+1,dtype=torch.long)

    L = 0
    for i in range(0,N):
        # Binary search for the largest positive j ≤ L such that X[M[j]] <= X[i]
        lo = 1
        hi = L
        while lo <= hi:
            mid = math.ceil((lo + hi) / 2)
            if X[M[mid]] <= X[i]:
                lo = mid + 1
            else:
                hi = mid - 1

        # After searching, lo is 1 greater than the length of the longest prefix of X[i]
        newL = lo

        # The predecessor of X[i] is the last index of the subsequence of length newL - 1
        P[i] = M[newL - 1]
        M[newL] = i  # save index of choosing element.

        if newL > L:
            # If we found a subsequence longer than any we've found yet, update L
            L = newL

    # Reconstruct the longest increasing subsequence LIS
    LIS = torch.zeros_like(inputTensor)
    k = M[L]
    for i in range(0,L):
        LIS[k] = X[k]
        k = P[k]

    return LIS
