# this network architecture inherit SurfaceSubnet_Q.

import sys
import torch

sys.path.append("../..")
from framework.NetTools import computeMuVariance, constructUnet
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import logits2Prob, MultiSurfaceCrossEntropyLoss
from framework.ConfigReader import ConfigReader
from OCTData.OCTDataUtilities import medianFilterSmoothing

class SurfaceSegNet_Q(BasicModel):
    def __init__(self, hps=None):
        '''
        inputSize: BxinputChaneels*H*W
        outputSize: (B, N, H, W)
        '''
        super().__init__()
        if isinstance(hps, str):
            hps = ConfigReader(hps)
        self.hps = hps
        C = self.hps.startFilters

        # input of Unet: BxinputChannelsxHxW
        self.m_downPoolings, self.m_downLayers, self.m_upSamples, self.m_upLayers = \
            constructUnet(self.hps.inputChannels, self.hps.inputHeight, self.hps.inputWidth, C, self.hps.nLayers)
        # output of Unet: BxCxHxW

        # Surface branch
        self.m_surfaces = nn.Sequential(
            Conv2dBlock(C, hps.segChannels),
            nn.Conv2d( hps.segChannels, self.hps.numSurfaces, kernel_size=1, stride=1, padding=0)  # conv 1*1
        )  # output size:BxNxHxW



    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        device = inputs.device
        # compute outputs
        skipxs = [None for _ in range(self.hps.nLayers)]  # skip link x

        # down path of Unet
        for i in range(self.hps.nLayers):
            if 0 == i:
                x = inputs
            else:
                x = skipxs[i-1]
            x = self.m_downPoolings[i](x)
            skipxs[i] = self.m_downLayers[i](x) + x

        # up path of Unet
        for i in range(self.hps.nLayers-1, -1, -1):
            if self.hps.nLayers-1 == i:
                x = skipxs[i]
            else:
                x = x+skipxs[i]
            x = self.m_upLayers[i](x) + x
            x = self.m_upSamples[i](x)
        # output of Unet: BxCxHxW


        # N is numSurfaces
        xs = self.m_surfaces(x)  # xs means x_surfaces, # output size: B*N*H*W
        B,N,H,W = xs.shape

        surfaceProb = logits2Prob(xs, dim=-2)

        # compute surface mu and variance
        mu, sigma2 = computeMuVariance(surfaceProb, layerMu=None, layerConf=None)  # size: B,N W

        if self.hps.useMedianFilterSmoothing:
            mu = medianFilterSmoothing(mu, winSize=self.hps.medianFilterWinSize)

        S = mu.clone()
        loss = 0.0
        if (self.getStatus() != "test") and self.hps.existGTLabel:
            multiSurfaceCEWeight = None
            if self.hps.useMultiSurfaceCEWeight:
                multiSurfaceCEWeight = torch.abs(inputs[:, 1, :, :])  # B,H,W,use GradH
                multiSurfaceCEWeight = logits2Prob(multiSurfaceCEWeight, dim=-2)  # along H normalization
                multiSurfaceCEWeight = multiSurfaceCEWeight.unsqueeze(dim=1)  # Bx1xHxW
                multiSurfaceCEWeight = multiSurfaceCEWeight.expand(xs.shape)  # BxNxHxW

            multiSurfaceCE = MultiSurfaceCrossEntropyLoss(weight=multiSurfaceCEWeight)
            loss_ce = multiSurfaceCE(surfaceProb, GTs)  # CrossEntropy is a kind of KLDiv

            l1Loss = nn.SmoothL1Loss().to(device)
            loss_L1 = l1Loss(mu, GTs)

            if "weightL1Loss" in self.hps.__dict__:
                weightL1 = self.hps.weightL1Loss
            else:
                weightL1 = 1.0

            loss =  loss_ce + loss_L1*weightL1

            #if torch.isnan(loss.sum()): # detect NaN
            #    print(f"Error: find NaN loss at epoch {self.m_epoch}")
            #    assert False

        if 1 == self.hps.hardSeparation:
             for i in range(1, N):
                S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :], S[:, i - 1, :], S[:, i, :])


        return S, sigma2, loss, x  # return surfaceLocation S in (B,S,W) dimension, sigma2, and loss, UnetFetures x.



