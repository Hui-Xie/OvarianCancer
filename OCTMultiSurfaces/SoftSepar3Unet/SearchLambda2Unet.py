# The main network for 2 unet for searching Lambda


import sys
import torch
import torch.optim as optim

sys.path.append("..")
from network.OCTOptimization import *
from network.QuadraticIPMOpt import SoftSeparationIPMModule
from network.OCTAugmentation import *

sys.path.append(".")
from SurfaceSubnet import SurfaceSubnet
from RiftSubnet import RiftSubnet

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.NetMgr import NetMgr
from framework.CustomizedLoss import SmoothSurfaceLoss, logits2Prob, WeightedDivLoss
from framework.ConfigReader import ConfigReader

class SearchLambda2Unet(BasicModel):
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
        self.m_sDevice = eval(self.hps.surfaceSubnetDevice)
        self.m_surfaceSubnet.to(self.m_sDevice)
        if "test" != surfaceMode:
            self.m_surfaceSubnet.setOptimizer(optim.Adam(self.m_surfaceSubnet.parameters(), lr=self.hps.surfaceSubnetLr, weight_decay=0))
            self.m_surfaceSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_surfaceSubnet.m_optimizer, \
                                            mode="min", factor=0.5, patience=20, min_lr=1e-8, threshold=0.02, threshold_mode='rel'))
        self.m_surfaceSubnet.setNetMgr(NetMgr(self.m_surfaceSubnet, self.m_surfaceSubnet.hps.netPath, self.m_sDevice))
        self.m_surfaceSubnet.m_netMgr.loadNet(surfaceMode)

        # rift Subnet
        self.m_riftSubnet = eval(self.hps.riftSubnet)(hps=self.hps.riftSubnetYaml)
        self.m_rDevice = eval(self.hps.riftSubnetDevice)
        self.m_riftSubnet.to(self.m_rDevice)
        if "test" != riftMode:
            self.m_riftSubnet.setOptimizer(
                optim.Adam(self.m_riftSubnet.parameters(), lr=self.hps.riftSubnetLr, weight_decay=0))
            self.m_riftSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_riftSubnet.m_optimizer, \
                                                                                 mode="min", factor=0.5, patience=20,
                                                                                 min_lr=1e-8, threshold=0.02,
                                                                                 threshold_mode='rel'))
        self.m_riftSubnet.setNetMgr(
            NetMgr(self.m_riftSubnet, self.m_riftSubnet.hps.netPath, self.m_rDevice))
        self.m_riftSubnet.m_netMgr.loadNet(riftMode)
        
        # lambda Subnet
        self.m_lDevice = eval(self.hps.lambdaSubnetDevice)


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


    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None, lambdaVec=None):
        Mu, Sigma2, surfaceLoss = self.m_surfaceSubnet.forward(inputs.to(self.m_sDevice),
                                     gaussianGTs=gaussianGTs.to(self.m_sDevice),
                                     GTs=GTs.to(self.m_sDevice))

        if 0 == self.hps.replaceRwithGT:  # 0: use predicted R;
            R, riftLoss = self.m_riftSubnet.forward(inputs.to(self.m_rDevice), gaussianGTs=None,GTs=None, layerGTs=None,
                                                riftGTs= riftGTs.to(self.m_rDevice))

        B,N,W = Mu.shape
        # expand Lambda into Bx(N-1)xW dimension
        Lambda = lambdaVec.view((1, (N-1), 1)).expand((B,(N-1),W)).to(self.m_lDevice)

        separationIPM = SoftSeparationIPMModule()
        # l1Loss = nn.SmoothL1Loss().to(self.m_lDevice)
        smoothSurfaceLoss = SmoothSurfaceLoss(mseLossWeight=10.0)

        assert (self.hps.status == "test") and (Lambda is not None)

        if 0 == self.hps.replaceRwithGT: # 0: use predicted R;
            R_detach = R.clone().detach().to(self.m_lDevice)
            print("use predicted R")
        elif 1 == self.hps.replaceRwithGT: #1: use riftGT without smoothness;
            R_detach = (GTs[:,1:, :] - GTs[:,0:-1, :]).detach().to(self.m_lDevice)
            print("use No-smooth ground truth R")
        elif 2 == self.hps.replaceRwithGT:  # 2: use smoothed riftGT;
            R_detach = riftGTs.clone().detach().to(self.m_lDevice)
            print("use smooth ground truth R")
        else:
            print(f"Wrong value of self.hps.replaceRwithGT")
            assert False

        Mu_detach = Mu.clone().detach().to(self.m_lDevice)
        Sigma2_detach = Sigma2.clone().detach().to(self.m_lDevice)
        Lambda_detach = Lambda.clone().detach().to(self.m_lDevice)

        S = separationIPM(Mu_detach, Sigma2_detach, R=R_detach, fixedPairWeight=self.hps.fixedPairWeight,
                          learningPairWeight=Lambda_detach)
        #surfaceL1Loss = l1Loss(S, GTs.to(self.m_lDevice))
        loss_smooth = smoothSurfaceLoss(S, GTs.to(self.m_lDevice))
        loss = loss_smooth


        return S, loss

    def zero_grad(self):
        if None != self.m_surfaceSubnet.m_optimizer:
            self.m_surfaceSubnet.m_optimizer.zero_grad()
        if None != self.m_riftSubnet.m_optimizer:
            self.m_riftSubnet.m_optimizer.zero_grad()
        if None != self.m_lambdaSubnet.m_optimizer:
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

