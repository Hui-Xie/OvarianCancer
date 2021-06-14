# Surface Subnet
# add mulitSurfaceCE Loss.
# erase Gaussian distribution force.
# first compute L1 loss, then ReLU.
# add Bidirectional Exponential Moving Average (BEMA)


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
from framework.CustomizedLoss import SmoothSurfaceLoss, logits2Prob, WeightedDivLoss, MultiSurfaceCrossEntropyLoss
from framework.ConfigReader import ConfigReader

class SurfaceSubnet_Q_BEMA(BasicModel):
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

        # define the BEMA(Bidirectional Exponential Moving Average) smooth matrix for IVUS circle surface.
        # S <-  S*M, each column of M denotes a BEMA.
        W = hps.inputWidth
        b = hps.BEMAMomentum  # momentum, beta, which denotes the previous weight of momentum.
        self.m_smoothM = torch.zeros((1, W, W), dtype=torch.float32, device=hps.device,  requires_grad=False)  #smooth matrix

        colM = torch.zeros((W, 1), dtype=torch.float32, device=hps.device,  requires_grad=False)  # a column of smooth matrix
        colM[0,0] = 1-b
        for i in range(1,W//2):
            colM[i,0] = 0.5*pow(b,i)*(1-b)
        for i in range(W//2, W):
            colM[i, 0] = 0.5 * pow(b, W-i) * (1 - b)

        for i in range(0, W):
            self.m_smoothM[0, :, i] = colM.roll(i).view(W)


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

        # smooth predicted S
        if self.hps.useBMEA:
            smoothM = self.m_smoothM.expand(B, W, W)  # size: BxWxW
            mu = torch.bmm(mu, smoothM)  # size: BxNxW

        S = mu.clone()
        loss = 0.0
        if (self.getStatus() != "test") and self.hps.existGTLabel:
            multiSurfaceCEWeight = None
            if self.hps.useMultiSurfaceCEWeight:
                multiSurfaceCEWeight =torch.abs(inputs[:, 1, :, :])  #B,H,W,use GradH
                multiSurfaceCEWeight = logits2Prob(multiSurfaceCEWeight, dim=-2) # along H normalization
                multiSurfaceCEWeight = multiSurfaceCEWeight.unsqueeze(dim=1)  #Bx1xHxW
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



