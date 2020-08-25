# The main network for 4 unet for soft separation


import sys
import torch
import torch.optim as optim

sys.path.append("..")
from network.OCTOptimization import *
from network.QuadraticIPMOpt import *
from network.OCTAugmentation import *

sys.path.append(".")
from SurfaceSubnet import SurfaceSubnet
from LambdaSubnet import LambdaSubnet

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.NetMgr import NetMgr
from framework.ConvBlocks import *
from framework.CustomizedLoss import SmoothSurfaceLoss, logits2Prob, WeightedDivLoss
from framework.ConfigReader import ConfigReader

class SoftSepar4Unet(BasicModel):
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

        NSurfaceMode, N_1Surface0Mode, N_1Surface1Mode, lambdaMode = self.getSubnetModes()

        # N-surface Subnet
        self.m_NSurfaceSubnet = eval(self.hps.NSurfaceSubnet)(hps=self.hps.NSurfaceSubnetYaml)
        self.m_NSurfaceDevice = eval(self.hps.NSurfaceSubnetDevice)
        self.m_NSurfaceSubnet.to(self.m_NSurfaceDevice)
        if "test" != NSurfaceMode:
            self.m_NSurfaceSubnet.setOptimizer(optim.Adam(self.m_NSurfaceSubnet.parameters(), lr=self.hps.NSurfaceSubnetLr, weight_decay=0))
            self.m_NSurfaceSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_NSurfaceSubnet.m_optimizer, \
                                                                                      mode="min", factor=0.5, patience=20, min_lr=1e-8, threshold=0.02, threshold_mode='rel'))
        self.m_NSurfaceSubnet.setNetMgr(NetMgr(self.m_NSurfaceSubnet, self.m_NSurfaceSubnet.hps.netPath, self.m_NSurfaceDevice))
        self.m_NSurfaceSubnet.m_netMgr.loadNet(NSurfaceMode)

        # (N-1)-surface0 Subnet
        self.m_N_1Surface0Subnet = eval(self.hps.N_1Surface0Subnet)(hps=self.hps.N_1Surface0SubnetYaml)
        self.m_N_1Surface0Device = eval(self.hps.N_1Surface0SubnetDevice)
        self.m_N_1Surface0Subnet.to(self.m_N_1Surface0Device)
        if "test" != N_1Surface0Mode:
            self.m_N_1Surface0Subnet.setOptimizer(
                optim.Adam(self.m_N_1Surface0Subnet.parameters(), lr=self.hps.N_1Surface0SubnetLr, weight_decay=0))
            self.m_N_1Surface0Subnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_N_1Surface0Subnet.m_optimizer, \
                                                                                      mode="min", factor=0.5,
                                                                                      patience=20, min_lr=1e-8,
                                                                                      threshold=0.02,
                                                                                      threshold_mode='rel'))
        self.m_N_1Surface0Subnet.setNetMgr(
            NetMgr(self.m_N_1Surface0Subnet, self.m_N_1Surface0Subnet.hps.netPath, self.m_N_1Surface0Device))
        self.m_N_1Surface0Subnet.m_netMgr.loadNet(N_1Surface0Mode)

        # (N-1)-surface1 Subnet
        self.m_N_1Surface1Subnet = eval(self.hps.N_1Surface1Subnet)(hps=self.hps.N_1Surface1SubnetYaml)
        self.m_N_1Surface1Device = eval(self.hps.N_1Surface1SubnetDevice)
        self.m_N_1Surface1Subnet.to(self.m_N_1Surface1Device)
        if "test" != N_1Surface1Mode:
            self.m_N_1Surface1Subnet.setOptimizer(
                optim.Adam(self.m_N_1Surface1Subnet.parameters(), lr=self.hps.N_1Surface1SubnetLr, weight_decay=0))
            self.m_N_1Surface1Subnet.setLrScheduler(
                optim.lr_scheduler.ReduceLROnPlateau(self.m_N_1Surface1Subnet.m_optimizer, \
                                                     mode="min", factor=0.5,
                                                     patience=20, min_lr=1e-8,
                                                     threshold=0.02,
                                                     threshold_mode='rel'))
        self.m_N_1Surface1Subnet.setNetMgr(
            NetMgr(self.m_N_1Surface1Subnet, self.m_N_1Surface1Subnet.hps.netPath, self.m_N_1Surface1Device))
        self.m_N_1Surface1Subnet.m_netMgr.loadNet(N_1Surface1Mode)

        # lambda Subnet
        self.m_lambdaSubnet = eval(self.hps.lambdaSubnet)(hps=self.hps.lambdaSubnetYaml)
        self.m_lDevice = eval(self.hps.lambdaSubnetDevice)
        self.m_lambdaSubnet.to(self.m_lDevice)
        if "test" != lambdaMode:
            self.m_lambdaSubnet.setOptimizer(
                optim.Adam(self.m_lambdaSubnet.parameters(), lr=self.hps.lambdaSubnetLr, weight_decay=0))
            self.m_lambdaSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_lambdaSubnet.m_optimizer, \
                                                                              mode="min", factor=0.5, patience=20,
                                                                              min_lr=1e-8, threshold=0.02,
                                                                              threshold_mode='rel'))
        self.m_lambdaSubnet.setNetMgr(
            NetMgr(self.m_lambdaSubnet, self.m_lambdaSubnet.hps.netPath, self.m_lDevice))
        self.m_lambdaSubnet.m_netMgr.loadNet(lambdaMode)

    def getSubnetModes(self):
        if self.hps.status == "trainLambda":
            NSurfaceMode = "test"
            N_1Surface0Mode = "test"
            N_1Surface1Mode = "test"
            lambdaMode = "train"
        elif self.hps.status == "fineTuneSurfaces":
            NSurfaceMode = "train"
            N_1Surface0Mode = "train"
            N_1Surface1Mode = "train"
            lambdaMode = "test"
        else:
            NSurfaceMode = "test"
            N_1Surface0Mode = "test"
            N_1Surface1Mode = "test"
            lambdaMode = "test"

        return NSurfaceMode, N_1Surface0Mode, N_1Surface1Mode, lambdaMode


    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        Mu, Sigma2, surfaceLoss = self.m_NSurfaceSubnet.forward(inputs.to(self.m_NSurfaceDevice),
                                                                gaussianGTs=gaussianGTs.to(self.m_NSurfaceDevice),
                                                                GTs=GTs.to(self.m_NSurfaceDevice))

        R, riftLoss = self.m_riftSubnet.forward(inputs.to(self.m_rDevice), gaussianGTs=None,GTs=None, layerGTs=None,
                                                riftGTs= riftGTs.to(self.m_rDevice))

        Lambda = self.m_lambdaSubnet.forward(inputs.to(self.m_lDevice))

        separationIPM = SoftSeparationIPMModule()
        # l1Loss = nn.SmoothL1Loss().to(self.m_lDevice)
        smoothSurfaceLoss = SmoothSurfaceLoss(mseLossWeight=10.0)

        if self.hps.status == "trainLambda":
            R_detach = R.clone().detach().to(self.m_lDevice)
            Mu_detach = Mu.clone().detach().to(self.m_lDevice)
            Sigma2_detach = Sigma2.clone().detach().to(self.m_lDevice)
            S = separationIPM(Mu_detach, Sigma2_detach, R=R_detach, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda)
            # surfaceL1Loss = l1Loss(S, GTs.to(self.m_lDevice))
            loss_smooth = smoothSurfaceLoss(S, GTs.to(self.m_lDevice))
            loss = loss_smooth

        elif self.hps.status == "fineTuneSurfaceRift":
            R = R.to(self.m_lDevice)
            Mu = Mu.to(self.m_lDevice)
            Sigma2 = Sigma2.to(self.m_lDevice)
            Lambda_detach = Lambda.clone().detach().to(self.m_lDevice)
            S = separationIPM(Mu, Sigma2, R=R, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda_detach)
            # surfaceL1Loss = l1Loss(S, GTs.to(self.m_lDevice))
            loss_smooth = smoothSurfaceLoss(S, GTs.to(self.m_lDevice))
            loss = surfaceLoss.to(self.m_lDevice) + riftLoss.to(self.m_lDevice) + loss_smooth

        elif self.hps.status == "test":
            R_detach = R.clone().detach().to(self.m_lDevice)
            Mu_detach = Mu.clone().detach().to(self.m_lDevice)
            Sigma2_detach = Sigma2.clone().detach().to(self.m_lDevice)
            Lambda_detach = Lambda.clone().detach().to(self.m_lDevice)
            S = separationIPM(Mu_detach, Sigma2_detach, R=R_detach, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda_detach)
            #surfaceL1Loss = l1Loss(S, GTs.to(self.m_lDevice))
            loss_smooth = smoothSurfaceLoss(S, GTs.to(self.m_lDevice))
            loss = loss_smooth

        else:
            assert False

        return S, loss

    def zero_grad(self):
        if None != self.m_NSurfaceSubnet.m_optimizer:
            self.m_NSurfaceSubnet.m_optimizer.zero_grad()
        if None != self.m_riftSubnet.m_optimizer:
            self.m_riftSubnet.m_optimizer.zero_grad()
        if None != self.m_lambdaSubnet.m_optimizer:
            self.m_lambdaSubnet.m_optimizer.zero_grad()


    def optimizerStep(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaSubnet.m_optimizer.step()
        elif self.hps.status == "fineTuneSurfaceRift":
            self.m_NSurfaceSubnet.m_optimizer.step()
            self.m_riftSubnet.m_optimizer.step()
        else:
            pass

    def lrSchedulerStep(self, validLoss):
        if self.hps.status == "trainLambda":
            self.m_lambdaSubnet.m_lrScheduler.step(validLoss)
        elif self.hps.status == "fineTuneSurfaceRift":
            self.m_NSurfaceSubnet.m_lrScheduler.step(validLoss)
            self.m_riftSubnet.m_lrScheduler.step(validLoss)
        else:
            pass

    def saveNet(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaSubnet.m_netMgr.saveNet()
        elif self.hps.status == "fineTuneSurfaceRift":
            self.m_NSurfaceSubnet.m_netMgr.saveNet()
            self.m_riftSubnet.m_netMgr.saveNet()
        else:
            pass

    def getLearningRate(self, subnetName):
        if subnetName == "lambdaSubnet":
            lr = self.m_lambdaSubnet.m_optimizer.param_groups[0]['lr']
        elif subnetName == "surfaceSubnet":
            lr = self.m_NSurfaceSubnet.m_optimizer.param_groups[0]['lr']
        elif subnetName == "riftSubnet":
            lr = self.m_riftSubnet.m_optimizer.param_groups[0]['lr']
        else:
            assert False

        return lr

