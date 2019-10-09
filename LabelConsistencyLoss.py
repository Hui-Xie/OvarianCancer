
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch
from scipy import ndimage
import numpy as np
import sys
import collections


class LabelConsistencyLoss(_Loss):
    "Current only support 3D volume"
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, windowSize=5, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda = lambdaCoeff
        self.m_windowSize= windowSize

    def forward(self, featureTensor, predictProb):
        assert featureTensor.ndim == predictProb.ndim
        N,_,X,Y,Z = featureTensor.size()
        ret = torch.tensor(0.0).to(featureTensor.device)
        m = self.m_windowSize//2  # margin

        # raw single thread  implement.
        visitedVoxels = collections.deque([() for _ in range((m+1)*Y*Z)])
        nCount = 0
        epsilon = 1e-8
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

        # parallel GPU implement
        # roll both featureTensor and predictProb, crop center, clip value, consine computation, sum, divided by 2.
        T1 = featureTensor[:,:, m:X-m, m:Y-m, m:Z-m]
        P1Full = predictProb[:,1, :, :, :]
        P1 = predictProb[:,1, m:X-m, m:Y-m, m:Z-m]
        for a in range(-m, m + 1):
            for b in range(-m, m + 1):
                for c in range(-m, m + 1):
                    if 0==a==b==c:
                        continue
                    T2 = torch.roll(featureTensor, (a,b,c), dim=(2,3,4))
                    P2 = torch.roll(P1Full, (a,b,c), dim=(2,3,4))


        return ret







