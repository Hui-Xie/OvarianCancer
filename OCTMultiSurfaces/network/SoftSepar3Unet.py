# The main network for 3 unet for soft separation


import sys
import torch
import torch.optim as optim

sys.path.append(".")
from OCTOptimization import *
from QuadraticIPMOpt import *
from OCTAugmentation import *
from SurfaceSubnet import SurfaceSubnet
from RiftSubnet import RiftSubnet
from LambdaSubnet import LambdaSubnet

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.NetMgr import NetMgr
from framework.ConvBlocks import *
from framework.CustomizedLoss import SmoothSurfaceLoss, logits2Prob, WeightedDivLoss

class SoftSepar3Unet(BasicModel):
    def __init__(self, hps=None):
        '''
        inputSize: BxinputChaneels*H*W
        outputSize: (B, N, W) surfaces
        '''
        super().__init__()
        self.hps = hps

        # Important:
        # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        # Parameters of a model after .cuda() will be different objects with those before the call.

        # surface Subnet
        self.m_surfaceSubnet = eval(self.hps.surfaceSubnet)(hps=self.hps.surfaceSubnetYaml)
        self.m_surfaceSubnet.to(device=self.hps.surfaceSubnetDevice)
        self.m_surfaceSubnet.setOptimizer(optim.Adam(self.m_surfaceSubnet.parameters(), lr=self.hps.surfaceSubnetLr, weight_decay=0))
        self.m_surfaceSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_surfaceSubnet.m_optimizer, \
                                            mode="min", factor=0.5, patience=20, min_lr=1e-8, threshold=0.02, threshold_mode='rel'))
        self.m_surfaceSubnet.setNetMgr(NetMgr(self.m_surfaceSubnet, self.m_surfaceSubnet.hps.netPath, self.hps.surfaceSubnetDevice))
        self.m_surfaceSubnet.m_netMgr.loadNet(self.hps.surfaceSubnetMode)

        # rift Subnet
        self.m_riftSubnet = eval(self.hps.riftSubnet)(hps=self.hps.riftSubnetYaml)
        self.m_riftSubnet.to(device=self.hps.riftSubnetDevice)
        self.m_riftSubnet.setOptimizer(
            optim.Adam(self.m_riftSubnet.parameters(), lr=self.hps.riftSubnetLr, weight_decay=0))
        self.m_riftSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_riftSubnet.m_optimizer, \
                                                                                 mode="min", factor=0.5, patience=20,
                                                                                 min_lr=1e-8, threshold=0.02,
                                                                                 threshold_mode='rel'))
        self.m_riftSubnet.setNetMgr(
            NetMgr(self.m_riftSubnet, self.m_riftSubnet.hps.netPath, self.hps.riftSubnetDevice))
        self.m_riftSubnet.m_netMgr.loadNet(self.hps.riftSubnetMode)
        
        # lambda Subnet
        self.m_lambdaSubnet = eval(self.hps.lambdaSubnet)(hps=self.hps.lambdaSubnetYaml)
        self.m_lambdaSubnet.to(device=self.hps.lambdaSubnetDevice)
        self.m_lambdaSubnet.setOptimizer(
            optim.Adam(self.m_lambdaSubnet.parameters(), lr=self.hps.lambdaSubnetLr, weight_decay=0))
        self.m_lambdaSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_lambdaSubnet.m_optimizer, \
                                                                              mode="min", factor=0.5, patience=20,
                                                                              min_lr=1e-8, threshold=0.02,
                                                                              threshold_mode='rel'))
        self.m_lambdaSubnet.setNetMgr(
            NetMgr(self.m_lambdaSubnet, self.m_lambdaSubnet.hps.netPath, self.hps.lambdaSubnetDevice))
        self.m_lambdaSubnet.m_netMgr.loadNet(self.hps.lambdaSubnetMode)





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
        S = mu.clone()
        for i in range(1, N):
            S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :], S[:, i - 1, :], S[:, i, :])

        loss_surface = 0.0
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
            loss_surface = weightedDivLoss(nn.LogSoftmax(dim=2)(xs), gaussianGTs)

            #hps.useSmoothSurface:
            smoothSurfaceLoss = SmoothSurfaceLoss(mseLossWeight=10.0)
            loss_smooth = smoothSurfaceLoss(S, GTs)

        loss = loss_surface + loss_smooth

        if torch.isnan(loss.sum()): # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return S, sigma2, loss  # return surfaceLocation S in (B,S,W) dimension, sigma2, and loss



