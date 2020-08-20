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
from framework.ConfigReader import ConfigReader

class SoftSepar3Unet(BasicModel):
    def __init__(self, hps=None):
        '''
        inputSize: BxinputChaneels*H*W
        outputSize: (B, N, W) surfaces
        '''
        super().__init__()
        if isinstance(hps, str):
            hps = ConfigReader(hps)
        self.hps = hps

        # Important:
        # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
        # Parameters of a model after .cuda() will be different objects with those before the call.

        surfaceMode, riftMode, lambdaMode = self.getSubnetModes()

        # surface Subnet
        self.m_surfaceSubnet = eval(self.hps.surfaceSubnet)(hps=self.hps.surfaceSubnetYaml)
        sDevice = eval(self.hps.surfaceSubnetDevice)
        self.m_surfaceSubnet.to(sDevice)
        self.m_surfaceSubnet.setOptimizer(optim.Adam(self.m_surfaceSubnet.parameters(), lr=self.hps.surfaceSubnetLr, weight_decay=0))
        self.m_surfaceSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_surfaceSubnet.m_optimizer, \
                                            mode="min", factor=0.5, patience=20, min_lr=1e-8, threshold=0.02, threshold_mode='rel'))
        self.m_surfaceSubnet.setNetMgr(NetMgr(self.m_surfaceSubnet, self.m_surfaceSubnet.hps.netPath, sDevice))
        self.m_surfaceSubnet.m_netMgr.loadNet(surfaceMode)

        # rift Subnet
        self.m_riftSubnet = eval(self.hps.riftSubnet)(hps=self.hps.riftSubnetYaml)
        rDevice = eval(self.hps.riftSubnetDevice)
        self.m_riftSubnet.to(rDevice)
        self.m_riftSubnet.setOptimizer(
            optim.Adam(self.m_riftSubnet.parameters(), lr=self.hps.riftSubnetLr, weight_decay=0))
        self.m_riftSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_riftSubnet.m_optimizer, \
                                                                                 mode="min", factor=0.5, patience=20,
                                                                                 min_lr=1e-8, threshold=0.02,
                                                                                 threshold_mode='rel'))
        self.m_riftSubnet.setNetMgr(
            NetMgr(self.m_riftSubnet, self.m_riftSubnet.hps.netPath, rDevice))
        self.m_riftSubnet.m_netMgr.loadNet(riftMode)
        
        # lambda Subnet
        self.m_lambdaSubnet = eval(self.hps.lambdaSubnet)(hps=self.hps.lambdaSubnetYaml)
        lDevice = eval(self.hps.lambdaSubnetDevice)
        self.m_lambdaSubnet.to(lDevice)
        self.m_lambdaSubnet.setOptimizer(
            optim.Adam(self.m_lambdaSubnet.parameters(), lr=self.hps.lambdaSubnetLr, weight_decay=0))
        self.m_lambdaSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_lambdaSubnet.m_optimizer, \
                                                                              mode="min", factor=0.5, patience=20,
                                                                              min_lr=1e-8, threshold=0.02,
                                                                              threshold_mode='rel'))
        self.m_lambdaSubnet.setNetMgr(
            NetMgr(self.m_lambdaSubnet, self.m_lambdaSubnet.hps.netPath, lDevice))
        self.m_lambdaSubnet.m_netMgr.loadNet(lambdaMode)

    def getSubnetModes(self):
        if self.hps.status == "trainLambda":
            surfaceMode = "test"
            riftMode = "test"
            lambdaMode = "train"
        elif self.hps.status == "fineTuneSurfaceRift":
            surfaceMode = "train"
            riftMode = "train"
            lambdaMode = "test"
        else:
            surfaceMode = "test"
            riftMode = "test"
            lambdaMode = "test"

        return surfaceMode, riftMode, lambdaMode


    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        sDevice = self.hps.surfaceSubnetDevice
        Mu, Sigma2, surfaceLoss = self.m_surfaceSubnet.forward(inputs.to(sDevice),
                                     gaussianGTs=gaussianGTs.to(sDevice),
                                     GTs=GTs.to(sDevice))

        rDevice = self.hps.riftSubnetDevice
        R, riftLoss = self.m_riftSubnet.forward(inputs.to(rDevice), gaussianGTs=None,GTs=None, layerGTs=None,
                                                riftGTs= riftGTs.to(rDevice))

        lDevice = self.hps.lambdaSubnetDevice
        Lambda = self.m_lambdaSubnet.forward(inputs.to(lDevice))

        separationIPM = SoftSeparationIPMModule()
        l1Loss = nn.SmoothL1Loss().to(lDevice)

        if self.hps.status == "trainLambda":
            R_detach = R.clone().detach().to(lDevice)
            Mu_detach = Mu.clone().detach().to(lDevice)
            Sigma2_detach = Sigma2.clone().detach().to(lDevice)
            S = separationIPM(Mu_detach, Sigma2_detach, R=R_detach, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda)
            surfaceL1Loss = l1Loss(S, GTs.to(lDevice))
            loss = surfaceL1Loss

        elif self.hps.status == "fineTuneSurfaceRift":
            R = R.to(lDevice)
            Mu = Mu.to(lDevice)
            Sigma2 = Sigma2.to(lDevice)
            Lambda_detach = Lambda.clone().detach().to(lDevice)
            S = separationIPM(Mu, Sigma2, R=R, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda_detach)
            surfaceL1Loss = l1Loss(S, GTs.to(lDevice))
            loss = surfaceLoss.to(lDevice) + riftLoss.to(lDevice) + surfaceL1Loss

        elif self.hps.status == "test":
            R_detach = R.clone().detach().to(lDevice)
            Mu_detach = Mu.clone().detach().to(lDevice)
            Sigma2_detach = Sigma2.clone().detach().to(lDevice)
            Lambda_detach = Lambda.clone().detach().to(lDevice)
            S = separationIPM(Mu_detach, Sigma2_detach, R=R_detach, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda_detach)
            surfaceL1Loss = l1Loss(S, GTs.to(lDevice))
            loss = surfaceL1Loss
        else:
            assert False

        return S, loss

    def zero_grad(self):
        self.m_surfaceSubnet.m_optimizer.zero_grad()
        self.m_riftSubnet.m_optimizer.zero_grad()
        self.m_lambdaSubnet.m_optimizer.zero_grad()


    def optimizerStep(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaSubnet.m_optimizer.step()
        elif self.hps.status == "fineTuneSurfaceRift":
            self.m_surfaceSubnet.m_optimizer.step()
            self.m_riftSubnet.m_optimizer.step()
        else:
            pass

    def lrSchedulerStep(self, validLoss):
        if self.hps.status == "trainLambda":
            self.m_lambdaSubnet.m_lrScheduler.step(validLoss)
        elif self.hps.status == "fineTuneSurfaceRift":
            self.m_surfaceSubnet.m_lrScheduler.step(validLoss)
            self.m_riftSubnet.m_lrScheduler.step(validLoss)
        else:
            pass

    def saveNet(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaSubnet.m_netMgr.saveNet()
        elif self.hps.status == "fineTuneSurfaceRift":
            self.m_surfaceSubnet.m_netMgr.saveNet()
            self.m_riftSubnet.m_netMgr.saveNet()
        else:
            pass

    def getLearningRate(self, subnetName):
        if subnetName == "lambdaSubnet":
            lr = self.m_lambdaSubnet.m_optimizer.param_groups[0]['lr']
        elif subnetName == "surfaceSubnet":
            lr = self.m_surfaceSubnet.m_optimizer.param_groups[0]['lr']
        elif subnetName == "riftSubnet":
            lr = self.m_riftSubnet.m_optimizer.param_groups[0]['lr']
        else:
            assert False

        return lr

