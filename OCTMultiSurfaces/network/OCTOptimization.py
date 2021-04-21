
# some optimizatino and measurement function for OCT Multisurface segmentation

import torch
import math

def computeMuVariance(x, layerMu=None, layerConf=None): # without square weight
    '''
    Compute the mean and variance along H direction of each surface.

    :param x: in (BatchSize, NumSurface, H, W) dimension, the value is probability (after Softmax) along each Height direction
           LayerMu: the referred surface mu from LayerProb, in size(B,N,W); where N = NumSurface.
           LayerConf: the referred surface confidence from LayerProb, in size(B,N,W)
    :return: mu:     mean in (BatchSize, NumSurface, W) dimension
             sigma2: variance in (BatchSize, Numsurface, W) dimension
    '''
    A =3.0  # weight factor to balance surfaceMu and LayerMu.

    device = x.device
    B,N,H,W = x.size() # Num is the num of surface for each patient

    # compute mu
    Y = torch.arange(H).view((1,1,H,1)).expand(x.size()).to(device=device, dtype=torch.int16)
    # mu = torch.sum(x*Y, dim=-2, keepdim=True)
    # use slice method to compute P*Y
    for b in range(B):
        if 0==b:
            PY = (x[b,]*Y[b,]).unsqueeze(dim=0)
        else:
            PY = torch.cat((PY, (x[b,]*Y[b,]).unsqueeze(dim=0)))
    mu = torch.sum(PY, dim=-2, keepdim=True) # size: B,N,1,W
    del PY  # hope to free memory.

    if (layerMu is not None) and (layerConf is not None):  # consider LayerMu, adjust mu computed by surface only
       assert layerMu.shape == layerConf.shape
       layerMu = layerMu.unsqueeze(dim=-2)
       layerConf = layerConf.unsqueeze(dim=-2)
       mu = (layerMu*layerConf + mu*(A-layerConf))/A

    # compute sigma2 (variance)
    Mu = mu.expand(x.size())

    #sigma2 = torch.sum(x*torch.pow(Y-Mu,2), dim=-2,keepdim=False)
    # this slice method is to avoid using big GPU memory .
    for b in range(B):
        if 0==b:
            sigma2 = torch.sum(x[b,]*torch.pow(Y[b,]-Mu[b,],2), dim=-2,keepdim=False).unsqueeze(dim=0)
        else:
            sigma2 = torch.cat((sigma2, torch.sum(x[b,]*torch.pow(Y[b,]-Mu[b,],2), dim=-2,keepdim=False).unsqueeze(dim=0)))

    # very important, otherwise sigma2 will increase to make the loss small
    # allowing sigma2 back propogation give better test result in the IVUS data.
    # todo: for experiment: /local/vol00/scratch/Users/hxie1/Projects/DeepLearningSeg/OCTMultiSurfaces/testConfig/
    #                      expUnetJHU_Surface_Layer_20200206/expUnetJHU_SurfaceNet_Sigma0_NoBPSigma_20200302_2.yaml
    # for IVUS data, Not backpropagating simga does not give better result;
    # At March 23rd, 2021, sigma2 should not backward in any optmization module.
    # At April 21st, 2021, allow sigma2 backward propagation. Application layers decide how to use sigma2.
    # At optiModel, opt variable is not sigma2, so it is not the case that optiModel will add simga2.
    # sigma2 = sigma2.detach()

    return mu.squeeze(dim=-2),sigma2


def computeMuVarianceWithSquare(x, layerMu=None, layerConf=None): # with square probability, then normalize
    '''
    Compute the mean and variance along H direction of each surface.

    :param x: in (BatchSize, NumSurface, H, W) dimension, the value is probability (after Softmax) along each Height direction
    :return: mu:     mean in (BatchSize, NumSurface, W) dimension
             sigma2: variance in (BatchSize, Numsurface, W) dimension
    '''
    A = 3.0  # weight factor to balance surfaceMu and LayerMu.
    device = x.device
    B, Num, H, W = x.size()  # Num is the num of surface for each patient

    # square probability to strengthen the big probability, and to reduce variance
    # "The rich get richer and the poor get poorer"
    P = torch.pow(x, 2).to(device=device,
                           dtype=torch.float32)  # it must use float32. float16 will lead sigma2 overflow(NaN)
    PColSum = torch.sum(P, dim=-2, keepdim=True).expand(P.size())  # column means H direction
    P = P / PColSum
    del PColSum  # in order to free memory for further reuse.

    # compute mu
    Y = torch.arange(H).view((1,1,H, 1)).expand(P.size()).to(device=device, dtype=torch.int16)
    # mu = torch.sum(P*Y, dim=-2, keepdim=True)
    # use slice method to comput P*Y
    for b in range(B):
        if 0 == b:
            PY = (P[b,] * Y[b,]).unsqueeze(dim=0)
        else:
            PY = torch.cat((PY, (P[b,] * Y[b,]).unsqueeze(dim=0)))
    mu = torch.sum(PY, dim=-2, keepdim=True) # size: B,N,1,W
    del PY  # hope to free memory.

    if (layerMu is not None) and (layerConf is not None):  # consider LayerMu, adjust mu computed by surface only
        assert layerMu.shape == layerConf.shape
        layerMu = layerMu.unsqueeze(dim=-2)
        layerConf = layerConf.unsqueeze(dim=-2)
        mu = (layerMu * layerConf + mu * (A - layerConf)) / A

    # compute sigma2 (variance)
    Mu = mu.expand(P.size())

    # sigma2 = torch.sum(P*torch.pow(Y-Mu,2), dim=-2,keepdim=False)
    # this slice method is to avoid using big GPU memory .
    for b in range(B):
        if 0 == b:
            sigma2 = torch.sum(P[b,] * torch.pow(Y[b,] - Mu[b,], 2), dim=-2, keepdim=False).unsqueeze(dim=0)
        else:
            sigma2 = torch.cat(
                (sigma2, torch.sum(P[b,] * torch.pow(Y[b,] - Mu[b,], 2), dim=-2, keepdim=False).unsqueeze(dim=0)))

    # very important, otherwise sigma2 will increase to make the loss small
    # all sigma2 back propogation give better test result in the IVUS data.
    # at March 23rd, 2021 sigma2 should not backward in any optmization module.
    # At April 21st, 2021, allow sigma2 backward propagation. Application layers decide how to use sigma2.
    # sigma2 = sigma2.detach()

    return mu.squeeze(dim=-2), sigma2







def getQFromVariance(sigma2):
    '''

    :param sigma2: variance in (BatchSize, N, W) dimension
    :return: Q: the diagonal reciprocal of variance in (B,W,N,N) size
    '''
    R = (1.0/sigma2).transpose(-1,-2) # the reciprocal in size of (B,W,N)
    Q = torch.diag_embed(R, offset=0)
    return Q


def proximalIPM(mu,sigma2, maxIterations=100, learningStep=0.01, criterion = 0.1 ):
    '''
    use proximal IPM method to optimize the final output surface location by Unet.
    It is used in inference stage.

    :param mu: mean of size (B,S,W), where S is surface
    :param sigma2: variance of size(B,S,W)
    :param nIterations: the iteration number  of proximal IPM method
    :return:
           S: the optimized surface locations in [B,S, W] dimension.
    '''
    # get initial sorted S0 in ascending order,
    batchLIS = getBatchLIS_gpu(mu)
    S0 = guaranteeSurfaceOrder(mu, batchLIS)
    if torch.all(mu.eq(S0)):
        return mu

    # S0 do not need gradient in back propagation
    S = S0.clone().detach()
    # IPM iteration
    for i in range(0, maxIterations):
        preS = S.clone()
        # S = S-learningStep*(S-mu)/sigma2  # 1st gradient method
        S = S-learningStep*(S-mu)         # newton's method in optimization
        batchLIS = getBatchLIS_gpu(S)
        S = guaranteeSurfaceOrder(S, batchLIS)
        if torch.abs(S-preS).mean() < criterion:
            break
    print(f"IPM used {i} iterations.")
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

    absError = torch.abs(predicitons-gts)
    stdSurface, muSurface = tuple(x*hPixelSize for x in torch.std_mean(absError, dim=(0,2)))
    stdPatient = torch.FloatTensor([torch.std(absError[i * slicesPerPatient:(i + 1) * slicesPerPatient, ]) for i in range(N // slicesPerPatient)]) *hPixelSize
    muPatient  = torch.FloatTensor([torch.mean(absError[i * slicesPerPatient:(i + 1) * slicesPerPatient, ]) for i in range(N // slicesPerPatient)]) *hPixelSize
    std, mu = tuple(x*hPixelSize for x in torch.std_mean(absError))
    return stdSurface, muSurface, stdPatient,muPatient, std,mu

def computeErrorStdMuOverPatientDimMean(predicitons, gts, slicesPerPatient=31, hPixelSize=3.870, goodBScansInGtOrder=None):
    '''
    Compute error standard deviation and mean along different dimension.

    First convert absError on patient dimension


    :param predicitons: in (BatchSize, NumSurface, W) dimension, in strictly patient order.
    :param gts: in (BatchSize, NumSurface, W) dimension
    :param hPixelSize: in micrometer
    :param goodBScansInGtOrder:
    :return: muSurface: (NumSurface) dimension, mean for each surface
             stdSurface: (NumSurface) dimension
             muPatient: (NumPatient) dimension, mean for each patient
             stdPatient: (NumPatient) dimension
             mu: a scalar, mean over all surfaces and all batchSize
             std: a scalar
    '''
    device = predicitons.device
    B,N, W = predicitons.shape # where N is numSurface
    absError = torch.abs(predicitons-gts)

    if goodBScansInGtOrder is None:
        P = B // slicesPerPatient
        absErrorPatient = torch.zeros((P,N), device=device)
        for p in range(P):
            absErrorPatient[p,:] = torch.mean(absError[p * slicesPerPatient:(p + 1) * slicesPerPatient, ], dim=(0,2))*hPixelSize
    else:
        P = len(goodBScansInGtOrder)
        absErrorPatient = torch.zeros((P, N), device=device)
        for p in range(P):
            absErrorPatient[p,:] = torch.mean(absError[p * slicesPerPatient+goodBScansInGtOrder[p][0]:p * slicesPerPatient+goodBScansInGtOrder[p][1], ], dim=(0,2))*hPixelSize

    stdSurface, muSurface = torch.std_mean(absErrorPatient, dim=0)
    # size of stdSurface, muSurface: [N]
    std, mu = torch.std_mean(absErrorPatient)
    return stdSurface, muSurface, std,mu

def computeSpecificSurfaceErrorForEachPatient(surfaceIndex, predicitons, gts, slicesPerPatient=31, hPixelSize=3.870, goodBScansInGtOrder=None):
    '''

    :param predicitons: in (BatchSize, NumSurface, W) dimension, in strictly patient order.
    :param gts: in (BatchSize, NumSurface, W) dimension
    :param hPixelSize: in micrometer
    :param goodBScansInGtOrder:
    :return: nPatients*1 vector
    '''
    device = predicitons.device
    B,N, W = predicitons.shape # where N is numSurface
    absError = torch.abs(predicitons-gts)

    if goodBScansInGtOrder is None:
        P = B // slicesPerPatient
        absErrorPatient = torch.zeros((P,N), device=device)
        for p in range(P):
            absErrorPatient[p,:] = torch.mean(absError[p * slicesPerPatient:(p + 1) * slicesPerPatient, ], dim=(0,2))*hPixelSize
    else:
        P = len(goodBScansInGtOrder)
        absErrorPatient = torch.zeros((P, N), device=device)
        for p in range(P):
            absErrorPatient[p,:] = torch.mean(absError[p * slicesPerPatient+goodBScansInGtOrder[p][0]:p * slicesPerPatient+goodBScansInGtOrder[p][1], ], dim=(0,2))*hPixelSize

    return absErrorPatient[:, surfaceIndex]

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

# OCT_DislocationLoss measures the dislocation between ground truth and the output location of dynamic programming
class OCT_DislocationLoss():
    def __init__(self, reduction='mean'):
        self.m_reduction = reduction

    def __call__(self, DPLoc, logP, GTs):
        epsilon = -1e-8
        logPGTs = logP.gather(2,GTs.long().unsqueeze(dim=2)).squeeze(dim=2)
        logPLoc = logP.gather(2,DPLoc.unsqueeze(dim=2)).squeeze(dim=2)

        loss = torch.pow(GTs-DPLoc, 2)*logPGTs/(epsilon+logPLoc)
        loss = loss.sum()
        if "mean" == self.m_reduction:
            num = torch.numel(GTs)
            loss /=num
        return loss

# may discard it, as it is a cpu version too slow.
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

# may depreciate it.
# the assumption that confusion region has same mu value, which is not optimal solution when sigmas in them have big difference.
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
# this can not guarantee the minimum cost:
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



def guaranteeSurfaceOrder(S, batchLIS):
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
    assert S.size() == batchLIS.size()
    B, surfaceNum, W = S.size()

    # get minimum value along column dimension in S
    columnMin, _ = torch.min(S, dim=-2)
    # for surface=0; if batachLIS=0, fill the above column minimum value
    batchLIS[:,0,:] = torch.where(0 == batchLIS[:,0,:], columnMin, batchLIS[:,0,:])
    for i in range(1,surfaceNum):
        batchLIS[:, i, :] = torch.where(0 == batchLIS[:,i,:], batchLIS[:, i - 1, :], batchLIS[:, i, :])
    return batchLIS

def getBatchLIS_cpu(mu):
    '''

    :param mu:
    :return: return a cpu-version batched LIS.
    '''
    B, surfaceNum, W = mu.size()
    LIS_cpu = torch.zeros(mu.size(), device='cpu', dtype=torch.float)
    mu_cpu = mu.cpu()
    for b in range(0,B):

       for w in range(0,W):
            LIS_cpu[b,:,w] = getaLIS(mu_cpu[b, :, w])
    return LIS_cpu

def getaLIS(inputTensor):
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

    M = torch.zeros(N+1,dtype=torch.long) #   stores the index k of the smallest value X[k]
    # such that there is an increasing subsequence of length j ending at X[k] on the range k ≤ i. Note that j ≤ (i+1),
    # because j ≥ 1 represents the length of the increasing subsequence, and k ≥ 0 represents the index of its termination.

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

def getBatchLIS_gpu(X):
    '''
    get Largest Increasing Subsequence  with non-choosing element marked as 0, in each H direction.
    for inpupt X of size (B,S,W)
    https://en.wikipedia.org/wiki/Longest_increasing_subsequence

    all index use torch.long

    :param X:  of size(B,N,W)
    :return: Tensor with choosing element in its original location keeping original value, and non-choosing element marked as 0, same length with input X
    '''

    B,S,W = X.shape
    device =X.device
    P = torch.zeros((B,S,W), dtype=torch.long, device=device)  #  stores the index of the predecessor of X[k] in the longest increasing subsequence ending at X[k].

    M = torch.zeros((B,S+1,W),dtype=torch.long, device=device)  #   stores the index k of the smallest value X[k]
    # such that there is an increasing subsequence of length j ending at X[k] on the range k ≤ i. Note that j ≤ (i+1),
    # because j ≥ 1 represents the length of the increasing subsequence, and k ≥ 0 represents the index of its termination.

    L =  torch.zeros((B,W), dtype=torch.long, device=device) # length of each LIS
    for i in range(0,S):
        # Binary search for the largest positive j ≤ L such that X[M[j]] <= X[i]
        lo = torch.ones((B,W), dtype=torch.long, device=device)
        hi = L
        mid = torch.zeros((B,W), dtype=torch.long, device=device)
        while (lo <= hi).any():  # here maybe dead loop, when sigma=0.
            mid = torch.where(lo <= hi, torch.ceil((lo + hi) / 2.0).long(), mid)
            mid3D = mid.unsqueeze(dim=1)
            X_M_mid = X.gather(1, M.gather(1,mid3D)).squeeze(dim=1)
            loRaw = lo.clone()
            lo  = torch.where((X_M_mid <= X[:,i,:]) & (lo <= hi), mid+1, lo)
            hi  = torch.where((X_M_mid >  X[:,i,:]) & (loRaw <= hi), mid-1, hi)


        # After searching, lo is 1 greater than the length of the longest prefix of X[i]
        newL = lo
        newL3D = newL.unsqueeze(dim=1)

        # The predecessor of X[i] is the last index of the subsequence of length newL - 1
        P[:, i, :] = M.gather(1,newL3D-1).squeeze(dim=1)
        M.scatter_(1, newL3D, i)

        # If we found a subsequence longer than any we've found yet, update L
        L = torch.where(newL>L, newL, L)


    # Reconstruct the longest increasing subsequence LIS
    LIS = torch.zeros_like(X)
    L3D = L.unsqueeze(dim=1)
    k = M.gather(1,L3D)  # k is a 3D tensor.
    while (L3D>0).any():
        LIS.scatter_(1, k, torch.where(L3D>0, X.gather(1, k), LIS.gather(1,k)))
        k = P.gather(1, k)
        L3D = L3D-1

    return LIS

def DPComputeSurfaces(logP):
    '''
    use Dynamic Programming to compute best possible path choosing along a column, which has maximum probability for all chosen
    locations.
    This computation do not need gradient because of its combination optimization.
    :param logP: in (B, NumSurfaces,H, W) size. each element indicates the log probability of this location belong to a surface.
    :return: Loc:  in (B,NumSurfaces, W) size with dtype=torch.long. return most possible surface locations.

    Notes: the real groundtruth for OCT surface are integers.
    '''

    device = logP.device
    with torch.no_grad():
        # build reward table for maximum probabilty path choosing.
        R = logP.clone().detach() # Reward table with original logP as all initial values
        B, NumSurfaces, H, W = R.shape

        cumMaxValue =  torch.zeros((B,NumSurfaces,H, W), dtype=torch.float32, device=device)
        cumMaxIndex =  torch.zeros((B,NumSurfaces,H, W), dtype=torch.int16, device=device)

        # build surface 0  cumMax along H dimension
        cumMaxValue[:, 0, 0, :] = R[:, 0, 0, :]
        # indicate the index along H dimension in cumMaxValue
        cumMaxIndex[:, :, 0, :] = torch.zeros((B,NumSurfaces,W), dtype=torch.int16, device=device)
        for h in range(1,H):
            cumMaxValue[:, 0, h,:] = torch.where(R[:, 0, h,:] >= cumMaxValue[:, 0, h-1,:], R[:, 0, h,:], cumMaxValue[:, 0, h-1,:] )
            cumMaxIndex[:, 0, h,:] = torch.where(R[:, 0, h,:] >= cumMaxValue[:, 0, h-1,:], h*torch.ones((B,W), dtype=torch.int16, device=device),  cumMaxIndex[:, 0, h-1,:] )

        for s in range(1, NumSurfaces):
            R[:,s,:,:] += cumMaxValue[:, s-1,:, :]  # core recursive formula

            cumMaxValue[:, s, 0, :] = R[:, s, 0, :]
            for h in range(1, H):
                cumMaxValue[:, s, h, :] = torch.where(R[:, s, h, :] >= cumMaxValue[:, s, h - 1, :], R[:, s, h, :], cumMaxValue[:, s, h - 1, :])
                cumMaxIndex[:, s, h, :] = torch.where(R[:, s, h, :] >= cumMaxValue[:, s, h - 1, :], h*torch.ones((B,W), dtype=torch.int16, device=device), cumMaxIndex[:, s, h - 1, :])

        #debug:
        #print(f"cumMaxValue = \n{cumMaxValue.squeeze()}")
        #print(f"cumMaxIndex = \n{cumMaxIndex.squeeze()}")
        #print(f"R = \n{R.squeeze()}")

        # find the maximum R at final surface s and backtrack
        Loc = torch.zeros((B,NumSurfaces,W),dtype=torch.long, device=device)
        loc = (cumMaxIndex[:,NumSurfaces-1, H-1,:]).long()
        for s in range(NumSurfaces-1, -1, -1):
            Loc[:,s,:] = loc.squeeze(dim=1)
            if s != 0:
                loc = cumMaxIndex[:, s-1,:,:].gather(1,Loc[:,s,:].unsqueeze(dim=1))

        return Loc






