# surfaceUnet + thicknessUnet + learning Lambda


import sys
import torch
import torch.optim as optim

sys.path.append("..")
from network.OCTOptimization import *
from network.QuadraticIPMOpt import *
from network.OCTAugmentation import *

sys.path.append(".")
from SurfaceSubnet_M import SurfaceSubnet_M
from ThicknessSubnet_M import ThicknessSubnet_M
from LambdaModule import LambdaModule

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.NetMgr import NetMgr
from framework.ConvBlocks import *
from framework.CustomizedLoss import SmoothSurfaceLoss, logits2Prob, WeightedDivLoss
from framework.ConfigReader import ConfigReader

class SoftSeparationNet_A(BasicModel):
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

        surfaceMode, thicknessMode, lambdaMode = self.getSubnetModes()

        # surface Subnet
        self.m_sDevice = eval(self.hps.surfaceSubnetDevice)
        self.m_surfaceSubnet = eval(self.hps.surfaceSubnet)(hps=self.hps.surfaceSubnetYaml)
        self.m_surfaceSubnet.to(self.m_sDevice)
        self.m_surfaceSubnet.m_optimizer = None
        if "test" != surfaceMode:
            self.m_surfaceSubnet.setOptimizer(optim.Adam(self.m_surfaceSubnet.parameters(), lr=self.hps.surfaceSubnetLr, weight_decay=0))
            self.m_surfaceSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_surfaceSubnet.m_optimizer, \
                                            mode="min", factor=0.5, patience=20, min_lr=1e-8, threshold=0.02, threshold_mode='rel'))
        self.m_surfaceSubnet.setNetMgr(NetMgr(self.m_surfaceSubnet, self.m_surfaceSubnet.hps.netPath, self.m_sDevice))
        self.m_surfaceSubnet.m_netMgr.loadNet(surfaceMode) # loadNet will load saved learning rate

        # thickness Subnet, where r means thickness
        self.m_rDevice = eval(self.hps.thicknessSubnetDevice)
        self.m_thicknessSubnet = eval(self.hps.thicknessSubnet)(hps=self.hps.thicknessSubnetYaml)
        self.m_thicknessSubnet.to(self.m_rDevice)
        self.m_thicknessSubnet.m_optimizer = None
        if "test" != thicknessMode:
            self.m_thicknessSubnet.setOptimizer(
                optim.Adam(self.m_thicknessSubnet.parameters(), lr=self.hps.thicknessSubnetLr, weight_decay=0))
            self.m_thicknessSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_thicknessSubnet.m_optimizer, \
                                                                                 mode="min", factor=0.5, patience=20,
                                                                                 min_lr=1e-8, threshold=0.02,
                                                                                 threshold_mode='rel'))
        self.m_thicknessSubnet.setNetMgr(NetMgr(self.m_thicknessSubnet, self.m_thicknessSubnet.hps.netPath, self.m_rDevice))
        self.m_thicknessSubnet.m_netMgr.loadNet(thicknessMode) # loadNet will load saved learning rate
        
        # lambda Module
        self.m_lDevice = eval(self.hps.lambdaModuleDevice)
        self.m_lambdaModule = eval(self.hps.lambdaModule)(self.m_surfaceSubnet.hps.startFilters+self.m_thicknessSubnet.startFilters,\
                                                          self.m_surfaceSubnet.hps.numSurfaces,\
                                                          self.m_surfaceSubnet.hps.inputHeight, \
                                                          self.m_surfaceSubnet.hps.inputWidth)
        self.m_lambdaModule.to(self.m_lDevice)
        self.m_lambdaModule.m_optimizer = None
        if "test" != lambdaMode:
            self.m_lambdaModule.setOptimizer(
                optim.Adam(self.m_lambdaModule.parameters(), lr=self.hps.lambdaModuleLr, weight_decay=0))
            self.m_lambdaModule.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_lambdaModule.m_optimizer, \
                                                                              mode="min", factor=0.5, patience=20,
                                                                              min_lr=1e-8, threshold=0.02,
                                                                              threshold_mode='rel'))
        self.m_lambdaModule.setNetMgr(
            NetMgr(self.m_lambdaModule, self.m_lambdaModule.hps.netPath, self.m_lDevice))
        self.m_lambdaModule.m_netMgr.loadNet(lambdaMode)


    def getSubnetModes(self):
        if self.hps.status == "trainLambda":
            surfaceMode = "test"
            thicknessMode = "test"
            lambdaMode = "train"
        elif self.hps.status == "fineTune":
            surfaceMode = "train"
            thicknessMode = "train"
            lambdaMode = "train"
        else:
            surfaceMode = "test"
            thicknessMode = "test"
            lambdaMode = "test"

        return surfaceMode, thicknessMode, lambdaMode


    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, thicknessGTs=None):
        Mu, Sigma2, surfaceLoss = self.m_surfaceSubnet.forward(inputs.to(self.m_sDevice),
                                     gaussianGTs=gaussianGTs.to(self.m_sDevice),
                                     GTs=GTs.to(self.m_sDevice))

        if 0 == self.hps.replaceRwithGT:  # 0: use predicted R;
            R, thicknessLoss = self.m_thicknessSubnet.forward(inputs.to(self.m_rDevice), gaussianGTs=None,GTs=None, layerGTs=None,
                                                thicknessGTs= thicknessGTs.to(self.m_rDevice))

        if self.hps.useFixedLambda:
            B, N, W = Mu.shape
            # expand Lambda into Bx(N-1)xW dimension
            Lambda = self.m_lambdaVec.view((1, (N - 1), 1)).expand((B, (N - 1), W)).to(self.m_lDevice)
        else:
            Lambda = self.m_lambdaModule.forward(inputs.to(self.m_lDevice))

        separationIPM = SoftSeparationIPMModule()
        l1Loss = nn.SmoothL1Loss().to(self.m_lDevice)
        # smoothSurfaceLoss = SmoothSurfaceLoss(mseLossWeight=10.0)

        if self.hps.status == "trainLambda":
            R_detach = R.clone().detach().to(self.m_lDevice)
            Mu_detach = Mu.clone().detach().to(self.m_lDevice)
            Sigma2_detach = Sigma2.clone().detach().to(self.m_lDevice)
            S = separationIPM(Mu_detach, Sigma2_detach, R=R_detach, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda)
            surfaceL1Loss = l1Loss(S, GTs.to(self.m_lDevice))
            #loss_smooth = smoothSurfaceLoss(S, GTs.to(self.m_lDevice))
            loss = surfaceL1Loss

        elif self.hps.status == "fineTuneSurfacethickness":
            R = R.to(self.m_lDevice)
            Mu = Mu.to(self.m_lDevice)
            Sigma2 = Sigma2.to(self.m_lDevice)
            Lambda_detach = Lambda.clone().detach().to(self.m_lDevice)
            S = separationIPM(Mu, Sigma2, R=R, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda_detach)
            surfaceL1Loss = l1Loss(S, GTs.to(self.m_lDevice))
            # loss_smooth = smoothSurfaceLoss(S, GTs.to(self.m_lDevice))
            # at final finetune stage, accurate R and Mu does not give benefits.
            loss = surfaceL1Loss

        elif self.hps.status == "test":
            if 0 == self.hps.replaceRwithGT: # 0: use predicted R;
                R_detach = R.clone().detach().to(self.m_lDevice)
                #print("use predicted R")
            elif 1 == self.hps.replaceRwithGT: #1: use thicknessGT without smoothness;
                R_detach = (GTs[:,1:, :] - GTs[:,0:-1, :]).detach().to(self.m_lDevice)
                #print("use No-smooth ground truth R")
            elif 2 == self.hps.replaceRwithGT:  # 2: use smoothed thicknessGT;
                R_detach = thicknessGTs.clone().detach().to(self.m_lDevice)
                #print("use smooth ground truth R")
            else:
                print(f"Wrong value of self.hps.replaceRwithGT")
                assert False

            Mu_detach = Mu.clone().detach().to(self.m_lDevice)
            Sigma2_detach = Sigma2.clone().detach().to(self.m_lDevice)
            Lambda_detach = Lambda.clone().detach().to(self.m_lDevice)
            
            if self.hps.debug== True:
                reciprocalTwoSigma2 = 1.0/Sigma2_detach
                print(f"reciprocalTwoSigma2.shape = {reciprocalTwoSigma2.shape}")
                print(f"mean of reciprocalTwoSigma2 = {torch.mean(reciprocalTwoSigma2, dim=[0, 2])}")
                print(f"min of reciprocalTwoSigma2  = {torch.min(torch.min(reciprocalTwoSigma2, dim=0)[0], dim=-1)}")
                print(f"max of reciprocalTwoSigma2  = {torch.max(torch.max(reciprocalTwoSigma2, dim=0)[0], dim=-1)}")

                print(f"Lambda_detach.shape = {Lambda_detach.shape}")
                print(f"mean of Lambda_detach = {torch.mean(Lambda_detach, dim=[0, 2])}")
                print(f"min of Lambda_detach  = {torch.min(torch.min(Lambda_detach, dim=0)[0], dim=-1)}")
                print(f"max of Lambda_detach  = {torch.max(torch.max(Lambda_detach, dim=0)[0], dim=-1)}")
            
            S = separationIPM(Mu_detach, Sigma2_detach, R=R_detach, fixedPairWeight=self.hps.fixedPairWeight,
                              learningPairWeight=Lambda_detach)
            surfaceL1Loss = l1Loss(S, GTs.to(self.m_lDevice))
            #loss_smooth = smoothSurfaceLoss(S, GTs.to(self.m_lDevice))
            loss = surfaceL1Loss

        else:
            assert False

        return S, loss

    def zero_grad(self):
        if None != self.m_surfaceSubnet.m_optimizer:
            self.m_surfaceSubnet.m_optimizer.zero_grad()
        if None != self.m_thicknessSubnet.m_optimizer:
            self.m_thicknessSubnet.m_optimizer.zero_grad()
        if (None != self.m_lambdaModule) and (None != self.m_lambdaModule.m_optimizer):
            self.m_lambdaModule.m_optimizer.zero_grad()


    def optimizerStep(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaModule.m_optimizer.step()
        elif self.hps.status == "fineTuneSurfacethickness":
            self.m_surfaceSubnet.m_optimizer.step()
            self.m_thicknessSubnet.m_optimizer.step()
        else:
            pass

    def lrSchedulerStep(self, validLoss):
        if self.hps.status == "trainLambda":
            self.m_lambdaModule.m_lrScheduler.step(validLoss)
        elif self.hps.status == "fineTuneSurfacethickness":
            self.m_surfaceSubnet.m_lrScheduler.step(validLoss)
            self.m_thicknessSubnet.m_lrScheduler.step(validLoss)
        else:
            pass

    def saveNet(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaModule.m_netMgr.saveNet()
        elif self.hps.status == "fineTuneSurfacethickness":
            self.m_surfaceSubnet.m_netMgr.saveNet()
            self.m_thicknessSubnet.m_netMgr.saveNet()
        else:
            pass

    def getLearningRate(self, subnetName):
        if subnetName == "lambdaModule":
            lr = self.m_lambdaModule.m_optimizer.param_groups[0]['lr']
        elif subnetName == "surfaceSubnet":
            lr = self.m_surfaceSubnet.m_optimizer.param_groups[0]['lr']
        elif subnetName == "thicknessSubnet":
            lr = self.m_thicknessSubnet.m_optimizer.param_groups[0]['lr']
        else:
            assert False

        return lr

