
# some optimizatino and measurement function for OCT Multisurface segmentation

import torch
import math

def computeMuVariance(x):
    '''
    Compute the mean and variance along H direction of each surface.

    :param x: in (BatchSize, NumSurface, H, W) dimension, the value is probability (after Softmax) along each Height direction
    :return: mu:     mean in (BatchSize, NumSurface, W) dimension
             sigma2: variance in (BatchSize, Numsurface, W) dimension, which will be datach from computation graph.
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

    # very important, otherwise sigma2 will increase to make the loss small
    sigma2 = sigma2.detach()

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
    S = sortedS.clone().detach()
    # IPM iteration
    for i in range(nIterations):
        S = S-learningStep*(S-mu)/sigma2
        S = gauranteeSurfaceOrder(S)
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

def fillGapOfLIS_gpu(batchLIS, S):
    '''
    0 in batchLIS mean non LIS elements

    :param batchLIS:
    :param S:
    :return:
    '''

    # get lowerbound of each 0 element


    # get upperbound of each 0 element

    # fill the gap



# may discard it
def fillGapOfLIS(batchLIS_cpu, mu):
    '''
    bounded nearest neighbour interpolation.

    :param batchLIS_cpu: 0 is non LIS element.
    :param mu:
    :return: batchLIS GPU version.
    '''
    assert batchLIS_cpu.size() == mu.size()
    B, surfaceNum, W = mu.size()
    device = mu.device

    # convert tensor to cpu
    mu_cpu = mu.cpu()

    # element-wise local nearest neighbor interpolation constrained in its lower boundary and upper boundary
    for b in range(0, B):
        for w in range(0, W):
            s = 0
            while (s < surfaceNum):
                if 0 == batchLIS_cpu[b, s, w]:
                    n = 1  # n continuous disorder predicted locations
                    while s + n < surfaceNum and 0 == batchLIS_cpu[b, s + n, w]:
                        n += 1
                    if s - 1 >= 0:
                        lowerBound = batchLIS_cpu[b, s - 1, w]
                    if s + n < surfaceNum:
                        upperbound = batchLIS_cpu[b, s + n, w]
                    for k in range(0, n):
                        if 0==s:
                            batchLIS_cpu[b, s+k, w] = upperbound
                        elif s-1>=0 and s+n< surfaceNum:
                            if mu_cpu[b, s+k, w] <= lowerBound:
                                batchLIS_cpu[b, s+k, w] = lowerBound
                            else: #mu_cpu[b, s+k , w] > lowerBound
                                batchLIS_cpu[b, s+k, w] = upperbound
                                lowerBound = upperbound   # avoid between 2 boundaries, there is first a big value, then a small value.
                        else:   # s==surfaceNum-1
                            batchLIS_cpu[b, s+k, w] = lowerBound
                    s = s + n
                else:
                    s += 1
    return  batchLIS_cpu.to(device)

def markConfusionSectionFromLIS(mu, batchLIS_cpu):
    '''

    :param mu:
    :param batchLIS_cpu: 0 marks non-choosing LIS elements. mu value corresponding 0 locations are always greater than previous choosing element.
    :return: a tensor with 0 marking the disorder section in cpu
              min(confusionRegion) > lower boundary
              max(confusionRegion) < upper boundary

    some example:
    test disroder region
    mu = tensor([ 1,  5,  3,  2,  6,  7,  9, 12])
    LIS = tensor([ 1,  0,  0,  2,  6,  7,  9, 12])
    disorderLIS =tensor([ 1,  0,  0,  0,  6,  7,  9, 12])

    test disroder region
    mu = tensor([ 1,  5,  3,  2,  6,  9,  8, 10, 12])
    LIS = tensor([ 1,  0,  0,  2,  6,  0,  8, 10, 12])
    disorderLIS =tensor([ 1,  0,  0,  0,  6,  0,  0, 10, 12])

    test disroder region
    mu = tensor([ 4,  5,  3,  2,  6,  9,  8, 10, 12])
    LIS = tensor([ 4,  5,  0,  0,  6,  0,  8, 10, 12])
    disorderLIS =tensor([ 0,  0,  0,  0,  6,  0,  0, 10, 12])

    test disroder region
    mu = tensor([ 5,  2,  3,  5,  6,  9,  8, 10, 12])
    LIS = tensor([ 0,  2,  3,  5,  6,  0,  8, 10, 12])
    disorderLIS =tensor([ 0,  0,  0,  0,  6,  0,  0, 10, 12])

    '''
    assert batchLIS_cpu.size() == mu.size()
    B, surfaceNum, W = mu.size()

    # convert tensor to cpu
    mu_cpu = mu.cpu()

    # element-wise check the maximum of mu in 0 section of LIS, if following LIS element< the maximum, change it into 0
    for b in range(0, B):
        for w in range(0, W):
            s = 0
            theMax = None
            while (s < surfaceNum):
                if 0 == batchLIS_cpu[b, s, w]:
                    if theMax is None:
                        theMax = mu_cpu[b, s, w]
                    else:
                        theMax = mu_cpu[b,s,w] if mu_cpu[b,s,w] > theMax else theMax
                    n = 1  # n continuous disorder predicted locations
                    while s + n < surfaceNum and 0 == batchLIS_cpu[b, s + n, w]:
                        theMax = mu_cpu[b, s + n, w] if mu_cpu[b, s + n, w] > theMax else theMax
                        n += 1
                    k = s+n
                    while k< surfaceNum and 0 != batchLIS_cpu[b,k,w]:
                        if  mu_cpu[b,k,w] <= theMax:
                            batchLIS_cpu[b, k, w] = 0
                        else:
                            break
                        k += 1
                    s = k
                else:
                    s += 1

    # element-wise check the minimum of mu in 0 section of LIS, if previous LIS element> the minimum, change it into 0
    # in reverse direction of H direction
    for b in range(0, B):
        for w in range(0, W):
            s = surfaceNum-1
            theMin = None
            while (s >= 0 ):
                if 0 == batchLIS_cpu[b, s, w]:
                    if theMin is None:
                        theMin = mu_cpu[b, s, w]
                    else:
                        theMin = mu_cpu[b,s,w] if mu_cpu[b,s,w] < theMin else theMin
                    n = 1  # n continuous disorder predicted locations
                    while s - n >= 0 and 0 == batchLIS_cpu[b, s - n, w]:
                        theMin = mu_cpu[b, s - n, w] if mu_cpu[b, s - n, w] < theMin else theMin
                        n += 1
                    k = s-n
                    while k>=0 and 0 != batchLIS_cpu[b,k,w]:
                        if  mu_cpu[b,k,w] >= theMin:
                            batchLIS_cpu[b, k, w] = 0
                        else:
                            break
                        k -= 1
                    s = k
                else:
                    s -= 1

    return batchLIS_cpu


#in continuous disorder section, the optimization value should be its sigma2-inverse-weighted average
# this can not gaurantee the minimum cost:
# for example: 1,5,2,3,9 when all variances are 1, has better solution: 1, 3.3, 3.3, 3.3, 9
#              1,5,2,3,9 when all variances are 1,100, 1,1,1, has better solution: 1, 2, 2, 3,9
def constraintOptimization(mu, sigma2, confusionLIS_cpu):
    '''
    idea:
            batchLIS_cpu = getBatchLIS(mu)  # cpu version
            confusionLIS_cpu = markConfusionSectionFromLIS(mu, batchLIS_cpu)
            S = constraintOptimization(mu, sigma2, confusionLIS_cpu) # result is at gpu same with mu

    :param mu:
    :param sigma2:
    :param confusionLIS_cpu:
    :return:
    '''

    device = mu.device
    # convert tensor to cpu
    sigma2_cpu = sigma2.cpu()
    mu_cpu = mu.cpu()
    S = confusionLIS_cpu

    if torch.all(mu_cpu.eq(S)):
        return mu
    B, surfaceNum, W = mu_cpu.size()

    # find constraint optimization location
    # method: along the H(surface)direction, for each confusion region,
    #          then the variance-inverse-weighted average as one averaged location should be one best approximate to the all orginal n disorder locations.
    #          this is a local thinking to achieve subpixel accuracy.

    # element-wise local constraint optimization
    for b in range(0, B):
        for w in range(0, W):
            s = 0
            while (s < surfaceNum):
                if 0 == S[b, s, w]:
                    n = 1  # n continuous disorder predicted locations
                    while s + n < surfaceNum and 0 == S[b, s + n, w]:
                        n += 1
                    if 1 == n:
                        print(f"n should not equal 1 for confusion region")
                        assert(False)
                    else:
                        numerator = 0.0
                        denominator = 0.0
                        for k in range(0, n):
                            numerator += mu_cpu[b, s + k, w] / sigma2_cpu[b, s + k, w]
                            denominator += 1.0 / sigma2_cpu[b, s + k, w]
                        x = numerator / denominator
                        for k in range(0, n):
                            S[b, s + k, w] = x
                    s = s + n
                else:
                    s += 1
    S = S.to(device)

    return S



def gauranteeSurfaceOrder(S, batchLIS):
    '''
    for example:
    S input = tensor([1, 5, 3, 2, 6, 7, 8, 9])
    S output = tensor([1, 2, 2, 2, 6, 7, 8, 9])

    S input = tensor([1, 3, 2, 5, 6, 7, 8, 9])
    S output = tensor([1, 2, 2, 5, 6, 7, 8, 9])

    S input = tensor([1, 4, 5, 2, 6, 7, 8, 9])
    S output = tensor([1, 4, 5, 5, 6, 7, 8, 9])

    :param S:
    :return:
    '''
    B,surfaceNum,W = S.size()

    # check global order at entry
    S0 = S[:, :-1, :]
    S1 = S[:, 1:, :]
    if (S1 >= S0).all():
        return S

    # use upper bound and lower bound to replace its current location value
    # assume surface 0 (the top surface) is correct, and it will be basis for following layer.
    # simple neighbor layer switch does not gaurantee global order: for example: 1 5 3 2 6 7 8 9
    for i in range(1,surfaceNum):
        S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :] and 0 == batchLIS[:,i,:], S[:, i - 1, :], S[:, i, :])
        if i != surfaceNum-1:
            S[:, i, :] = torch.where(S[:, i, :] > S[:, i + 1, :] and 0 == batchLIS[:,i,:], S[:, i + 1, :], S[:, i, :])

    # recursive call to make sure order gaurantee
    S = gauranteeSurfaceOrder(S, batchLIS)
    return S

def getBatchLIS(mu):
    '''

    :param mu:
    :return: return a cpu-version batched LIS.
    '''
    B, surfaceNum, W = mu.size()
    LIS_cpu = torch.zeros(mu.size(), device='cpu', dtype=torch.float)
    mu_cpu = mu.cpu()
    for b in range(0,B):
        for w in range(0,W):
            LIS_cpu[b,:,w] = getLIS(mu_cpu[b,:,w])
    return LIS_cpu

def getLIS(inputTensor):
    '''
    get Largest Increasing Subsequence  with non-choosing element marked as 0
    https://en.wikipedia.org/wiki/Longest_increasing_subsequence
    for example:
    mu = tensor([ 4,  5,  3,  2,  6,  9,  8, 10, 12])
    LIS = tensor([ 4,  5,  0,  0,  6,  0,  8, 10, 12])

    :param inputTensor:
    :return: Tensor with choosing element in its location with non-choosing element marked as 0, same length with inputTensor
    '''
    X = inputTensor
    N = len(inputTensor)
    assert 1 == X.dim()
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

def getLIS_gpu(X):
    '''
    get Largest Increasing Subsequence  with non-choosing element marked as 0, in each H direction.
    https://en.wikipedia.org/wiki/Longest_increasing_subsequence



    :param X:  of size(B,S,W)
    :return: Tensor with choosing element in its location with non-choosing element marked as 0, same length with input X
    '''

    B,S,W = X.shape
    device =X.device
    P = torch.zeros((B,S,W), dtype=torch.long, device=device)  #  stores the index of the predecessor of X[k] in the longest increasing subsequence ending at X[k].
    M = torch.zeros((B,S+1,W),dtype=torch.long,device=device)  # maximum element in each active subsequence

    L =  torch.zeros((B,W), dtype=torch.long, device=device) # length of each LIS
    for i in range(0,S):
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
    LIS = torch.zeros_like(X)
    k = M[L]
    for i in range(0,L):
        LIS[k] = X[k]
        k = P[k]

    return LIS