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
from torch import linalg as LA

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

        # define the constant matrix B of size NxN and C of size Nx(N-1)
        N = self.m_surfaceSubnet.hps.numSurfaces
        self.m_B = torch.zeros((1, N, N), dtype=torch.float32, device=self.m_lDevice, requires_grad=False)
        self.m_B[0, 0, 1] = 1.0
        self.m_B[0, N-1, N-2] = 1.0
        for i in range(1, N - 1):
            self.m_B[0, i, i-1] = 0.5
            self.m_B[0, i, i+1] = 0.5

        self.m_C = torch.zeros((1, N, N-1), dtype=torch.float32, device=self.m_lDevice, requires_grad=False)
        self.m_C[0, 0, 0] = -1.0
        self.m_C[0, N - 1, N - 1] = 1.0
        for i in range(1, N - 1):
            self.m_C[0, i, i - 1] = 0.5
            self.m_C[0, i, i] = -0.5

        self.m_smoothM = self.m_surfaceSubnet.m_smoothM.to(self.m_lDevice)

        self.m_A = torch.zeros((1, N-1, N), dtype=torch.float32, device=self.m_lDevice, requires_grad=False)
        for i in range(0, N - 1):
            self.m_A[0, i, i] = 1.0
            self.m_A[0, i, i+1] = -1.0

        W = self.m_surfaceSubnet.inputWidth
        self.m_D = torch.zeros((1, W, W), dtype=torch.float32, device=self.m_lDevice, requires_grad=False)
        # 0th column and W-1 column
        self.m_D[0, 0, 0] = -25
        self.m_D[0, 1, 0] = 48
        self.m_D[0, 2, 0] = -36
        self.m_D[0, 3, 0] = 16
        self.m_D[0, 4, 0] = -3
        self.m_D[0,1:6,1] = self.m_D[0,0:5,0]

        self.m_D[0, W-5, W-1] = 3
        self.m_D[0, W-4, W-1] = -16
        self.m_D[0, W-3, W-1] = 36
        self.m_D[0, W-2, W-1] = -48
        self.m_D[0, W-1, W-1] = 25
        self.m_D[0, W - 6: W-1, W - 2]  = self.m_D[0, W-5:W, W-1]
        # columns from 2 to W-2
        for i in range(2, W - 2):
            self.m_D[0, i - 2, i] = 1.0
            self.m_D[0, i - 1, i] = -8.0
            self.m_D[0, i + 1, i] = 8.0
            self.m_D[0, i + 2, i] = -1.0

        self.m_alpha  = hps.alpha

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
        Mu, Sigma2, surfaceLoss, surfaceX = self.m_surfaceSubnet.forward(inputs.to(self.m_sDevice),
                                     gaussianGTs=gaussianGTs.to(self.m_sDevice),
                                     GTs=GTs.to(self.m_sDevice))

        # input channels: raw+Y+X  # todo
        assert(False)
        R, thicknessLoss, thinknessX = self.m_thicknessSubnet.forward(inputs.to(self.m_rDevice), gaussianGTs=None,GTs=None, layerGTs=layerGTs.to(self.m_rDevice),
                                                thicknessGTs= thicknessGTs.to(self.m_rDevice))

        X = torch.cat((surfaceX.to(self.m_lDevice), thinknessX.to(self.m_lDevice)), dim=1)
        Lambda = self.m_lambdaModule.forward(X)

        B,C,H,W = X.shape
        N = self.m_surfaceSubnet.hps.numSurfaces
        B = self.m_B.expand(B, N, N)
        C = self.m_C.expand(B, N, N - 1)
        M = self.m_smoothM.expand(B, W, W)
        A = self.m_A.expand(B,N-1,N)
        D = self.m_D.expand(B, W, W)

        G = GTs.to(self.m_lDevice)
        Q = (1.0/Sigma2).to(self.m_lDevice)

        #separationIPM = SoftSeparationIPMModule()
        #l1Loss = nn.SmoothL1Loss().to(self.m_lDevice)
        # smoothSurfaceLoss = SmoothSurfaceLoss(mseLossWeight=10.0)

        if self.hps.status == "trainLambda":
            R_detach = R.clone().detach().to(self.m_lDevice)
            Mu_detach = Mu.clone().detach().to(self.m_lDevice)
            Sigma2_detach = Sigma2.clone().detach().to(self.m_lDevice)
            S = torch.bmm(Lambda*Mu_detach+(1.0-Lambda)*(torch.bmm(B, Mu_detach)+torch.bmm(C,R_detach)), M)
            for i in range(1, N):  #ReLU
                S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :], S[:, i - 1, :], S[:, i, :])
            Unary = (S - G) * Q
            Pair = self.m_alpha*torch.bmm(R_detach+torch.bmm(A,S),D)
            loss = torch.mean(LA.norm(Unary,ord='fro', dim=(1,2), keepdim=False) \
                             +LA.norm(Pair,ord='fro', dim=(1,2), keepdim=False))  # size: B-> scalar

        elif self.hps.status == "fineTune":
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
        elif self.hps.status == "fineTune":
            self.m_surfaceSubnet.m_optimizer.step()
            self.m_thicknessSubnet.m_optimizer.step()
            self.m_lambdaModule.m_optimizer.step()
        else:
            pass

    def lrSchedulerStep(self, validLoss):
        if self.hps.status == "trainLambda":
            self.m_lambdaModule.m_lrScheduler.step(validLoss)
        elif self.hps.status == "fineTune":
            self.m_surfaceSubnet.m_lrScheduler.step(validLoss)
            self.m_thicknessSubnet.m_lrScheduler.step(validLoss)
            self.m_lambdaModule.m_lrScheduler.step(validLoss)
        else:
            pass

    def saveNet(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaModule.m_netMgr.saveNet()
        elif self.hps.status == "fineTune":
            self.m_surfaceSubnet.m_netMgr.saveNet()
            self.m_thicknessSubnet.m_netMgr.saveNet()
            self.m_lambdaModule.m_netMgr.saveNet()
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

