
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch
from scipy import ndimage
import numpy as np
import sys
import collections


class ConsistencyLoss1(_Loss):
    "Current only support 3D volume"
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, windowSize=5, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda = lambdaCoeff
        assert 1 == windowSize%2
        self.m_windowSize= windowSize

    def forward(self, featureTensor, predictProb):
        assert featureTensor.ndim == predictProb.ndim
        N,_,X,Y,Z = featureTensor.size()
        ret = torch.tensor(0.0).to(featureTensor.device)
        m = self.m_windowSize//2  # margin
        epsilon = 1e-8

        # raw single thread  implement. It is very very very slow.
        """
        visitedVoxels = collections.deque([() for _ in range((m+1)*Y*Z)])
        nCount = 0
        for n in range(N):
            for x in range(m, X-m):
                for y in range(m,Y-m):
                    for z in range(m, Z-m):
                        index1 = (x,y,z)
                        visitedVoxels.pop()
                        visitedVoxels.appendleft(index1)
                        v1 = featureTensor[n,:,x,y,z]
                        p1 = predictProb[n,1, x,y,z]
                        for a in range(-m,m+1):
                            for b in range(-m, m+1):
                                for c in range(-m, m+1):
                                    xx, yy, zz = x+a, y+b, z+c
                                    index2 = (xx,yy,zz)
                                    if index2 in visitedVoxels:
                                        continue
                                    v2 = featureTensor[n,:,xx,yy,zz]
                                    p2 = predictProb[n,1, xx,yy,zz]
                                    cosineSm = F.cosine_similarity(v1,v2, dim=0)
                                    ftrDiff = (1 -cosineSm)/2  # feature difference between v1 and v2 feature vectors
                                    p12= p1-p2 if p1>=p2 else p2-p1  # predicted prob difference
                                    if p12==0:
                                        p12 = p12+ epsilon
                                    if p12 ==1:
                                        p12 = p12- epsilon
                                    ret += -ftrDiff*torch.log(p12)-(1-ftrDiff)*torch.log(1-p12)
                                    nCount +=1

        ret = ret/nCount*self.m_lambda
         """

        # parallel GPU implement
        # roll both featureTensor and predictProb, crop center, clip value, consine computation, sum, divided by 2.
        nCount = 0
        T1 = featureTensor[:,:, m:X-m, m:Y-m, m:Z-m]
        P1Full = predictProb[:,1, :, :, :]  # Now P1Full becomes 4D tensor (N, X, Y, Z).
        P1 = predictProb[:,1, m:X-m, m:Y-m, m:Z-m]   #4D tensor (N, X, Y, Z).
        for a in range(-m, m + 1):   # only shift half nodes around center node
            for b in range(-m, a + 1):
                for c in range(-m, m + 1):
                    if (a == b and a > 0) or (a == b == 0 and c >= 0):
                        continue
                    T2 = torch.roll(featureTensor, (a,b,c), dims=(2,3,4))
                    P2 = torch.roll(P1Full, (a,b,c), dims=(1,2,3))
                    T2 = T2[:,:, m:X-m, m:Y-m, m:Z-m]
                    P2 = P2[:,m:X-m, m:Y-m, m:Z-m]
                    TSimilarity = F.cosine_similarity(T1,T2, dim=1)  # now it is 4D tensor
                    TDiff = (1.0- TSimilarity)/2.0   # feature difference between T1 and T2 feature Tensors
                    P12 = torch.abs(P1-P2)           # predicted prob difference
                    P12 = torch.clamp(P12, epsilon, 1.0-epsilon)
                    ret += torch.sum(-TDiff * torch.log(P12) - (1.0 - TDiff) * torch.log(1.0 - P12))
                    nCount +=1

        ret = ret/(nCount*P1.numel())

        return ret



# use groundtruth, instead of internal predict.
class ConsistencyLoss2(_Loss):
    "Current only support 3D volume"
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, windowSize=5, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda = lambdaCoeff
        assert 1 == windowSize%2
        self.m_windowSize= windowSize

    def forward(self, featureTensor, gts):
        N,_,X,Y,Z = featureTensor.size()
        ret = torch.tensor(0.0).to(featureTensor.device)
        m = self.m_windowSize//2  # margin
        epsilon = 1e-8

        # parallel GPU implement
        # roll both featureTensor and predictProb, crop center, clip value, consine computation, sum, divided by 2.
        nCount = 0
        T1 = featureTensor[:,:, m:X-m, m:Y-m, m:Z-m]
        G1 = gts[:,m:X-m, m:Y-m, m:Z-m]   #4D tensor (N, X, Y, Z).
        for a in range(-m, m + 1):   # only shift half nodes around center node
            for b in range(-m, a + 1):
                for c in range(-m, m + 1):
                    if (a == b and a > 0) or (a == b == 0 and c >= 0):
                        continue
                    T2 = torch.roll(featureTensor, (a,b,c), dims=(2,3,4))
                    G2 = torch.roll(gts, (a,b,c), dims=(1,2,3))
                    T2 = T2[:,:, m:X-m, m:Y-m, m:Z-m]
                    G2 = G2[:,m:X-m, m:Y-m, m:Z-m]
                    TSimilarity = F.cosine_similarity(T1,T2, dim=1)  # now it is 4D tensor
                    TDiff = (1.0- TSimilarity)/2.0   # feature difference between T1 and T2 feature Tensors
                    G12 = torch.abs(G1-G2).float()           # groundtruth prob difference
                    TDiff = torch.clamp(TDiff, epsilon, 1.0-epsilon)
                    ret += torch.sum(-G12 * torch.log(TDiff) - (1.0 - G12) * torch.log(1.0 - TDiff))
                    nCount +=1

        ret = ret/(nCount*G1.numel())

        return ret





