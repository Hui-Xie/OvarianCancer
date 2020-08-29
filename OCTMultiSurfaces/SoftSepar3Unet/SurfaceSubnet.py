# Surface Subnet


import sys
import torch

sys.path.append("..")
from network.OCTOptimization import *
from network.OCTAugmentation import *
from network.QuadraticIPMOpt import SoftSeparationIPMModule

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import SmoothSurfaceLoss, logits2Prob, WeightedDivLoss
from framework.ConfigReader import ConfigReader

class SurfaceSubnet(BasicModel):
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
            Conv2dBlock(C, C),
            nn.Conv2d(C, self.hps.numSurfaces, kernel_size=1, stride=1, padding=0)  # conv 1*1
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

        layerMu = None # referred surface mu computed by layer segmentation.
        layerConf = None
        surfaceProb = logits2Prob(xs, dim=-2)

        # compute surface mu and variance
        mu, sigma2 = computeMuVariance(surfaceProb, layerMu=layerMu, layerConf=layerConf)  # size: B,N W

        # ReLU to guarantee layer order not to cross each other
        if self.hps.hardSeparation:
            separationIPM = SoftSeparationIPMModule()
            S = separationIPM(mu, sigma2, R=None, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=None) # only use unary item
        else:
            S = mu.clone()
            for i in range(1, N):
                S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :], S[:, i - 1, :], S[:, i, :])


        loss_div = 0.0
        loss_smooth = 0.0
        if self.hps.existGTLabel:
            # hps.useWeightedDivLoss:
            surfaceWeight = None
            _, C, _, _ = inputs.shape
            if C >= 4:  # at least 3 gradient channels.
                imageGradMagnitude = inputs[:, C - 1, :,
                                     :]  # image gradient magnitude is at final channel since July 23th, 2020
                surfaceWeight = getSurfaceWeightFromImageGradient(imageGradMagnitude, N, gradWeight=self.hps.gradWeight)

            weightedDivLoss = WeightedDivLoss(weight=surfaceWeight ) # the input given is expected to contain log-probabilities
            if 0 == len(gaussianGTs):  # sigma ==0 case
                gaussianGTs = batchGaussianizeLabels(GTs, sigma2, H)
            loss_div = weightedDivLoss(nn.LogSoftmax(dim=2)(xs), gaussianGTs)

            #hps.useSmoothSurface:
            smoothSurfaceLoss = SmoothSurfaceLoss(mseLossWeight=10.0)
            loss_smooth = smoothSurfaceLoss(S, GTs)

        loss = loss_div + loss_smooth

        if torch.isnan(loss.sum()): # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return S, sigma2, loss  # return surfaceLocation S in (B,S,W) dimension, sigma2, and loss



