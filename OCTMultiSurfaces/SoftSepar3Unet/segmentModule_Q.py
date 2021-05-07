# Learninng Lambda module
# train on all training data, and validation on validation data

import sys
import torch

sys.path.append("..")
from network.OCTOptimization import *
from network.OCTAugmentation import *

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import  logits2Prob
from framework.ConfigReader import ConfigReader

from framework.CustomizedLoss import logits2Prob, MultiSurfaceCrossEntropyLoss

class segmentModule_Q(BasicModel):
    def __init__(self, inputC,segC, hps=None):
        '''
        inputSize: BxinputCxHxW
        inputC: the number of input channels
        segC: seg block channels.
        N: the number of surfaces
        H: image height
        W: image width
        outputSize: (B, N, W) in [0,1]
        '''
        super().__init__()

        self.hps = hps
        N = hps.numSurfaces

        self.m_mergeSurfaces = nn.Sequential(
            Conv2dBlock(inputC, segC),
            nn.Conv2d(segC, N, kernel_size=1, stride=1, padding=0)  # conv 1*1
        )  # output size:BxNxHxW

    def copyWeightFrom(self, surfaceSeq, thickSeq):
        assert (len(surfaceSeq) == len(thickSeq))
        device = self.hps.device
        with torch.no_grad():
            self.m_mergeSurfaces[0].m_conv.weight.data = torch.cat((surfaceSeq[0].m_conv.weight.data.to(device),
                                                               thickSeq[0].m_conv.weight.data.to(device)), dim=1)
            self.m_mergeSurfaces[0].m_conv.bias.data = 0.5*(surfaceSeq[0].m_conv.bias.data.to(device)+thickSeq[0].m_conv.bias.data.to(device))
            self.m_mergeSurfaces[1].weight.data = 0.5*(surfaceSeq[1].weight.data.to(device)+ thickSeq[1].weight.data.to(device))
            self.m_mergeSurfaces[1].bias.data = 0.5 * (surfaceSeq[1].bias.data.to(device) + thickSeq[1].bias.data.to(device))

    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        # N is numSurfaces
        device = inputs.device
        xs = self.m_mergeSurfaces(inputs)  # xs means x_surfaces, # output size: B*N*H*W
        B, N, H, W = xs.shape

        surfaceProb = logits2Prob(xs, dim=-2)

        # compute surface mu and variance
        mu, sigma2 = computeMuVariance(surfaceProb, layerMu=None, layerConf=None)  # size: B,N W

        S = mu.clone()
        loss = 0.0
        if (self.getStatus() != "test") and self.hps.existGTLabel:
            multiSurfaceCE = MultiSurfaceCrossEntropyLoss()
            loss_ce = multiSurfaceCE(surfaceProb, GTs)  # CrossEntropy is a kind of KLDiv

            l1Loss = nn.SmoothL1Loss().to(device)
            loss_L1 = l1Loss(mu, GTs)

            loss = loss_ce + loss_L1

        if 1 == self.hps.hardSeparation:
            for i in range(1, N):
                S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :], S[:, i - 1, :], S[:, i, :])

        return S, sigma2, loss  # return surfaceLocation S in (B,S,W) dimension, sigma2, and loss, UnetFetures x.
