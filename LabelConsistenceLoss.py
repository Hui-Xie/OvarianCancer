
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch
from scipy import ndimage
import numpy as np
import sys
import collections


class LabelConsistenceLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, windowSize=5, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_windowSize= windowSize

    def forward(self, featureTensor, predictProb):
        assert featureTensor.dim == predictProb.dim
        N,C,X,Y,Z = featureTensor.size()
        ret = torch.zeros(N).to(featureTensor.device)
        m = self.m_windowSize//2  # margin

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
                                    cosineSm = F.cosine_similarity(v1,v2)
                                    y = (1 -cosineSm)/2  # feature difference between v1 and v2 feature vectors
                                    p12= p1-p2 if p1>=p2 else p2-p1  # predicted prob difference
                                    if p12==0:
                                        p12 = p12+ epsilon
                                    ret += -y*torch.log(p12)







