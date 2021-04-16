# surfaceUnet + thicknessUnet + learning Lambda
# Model 3: NXW learning lambda, Smooth, Gradient pairwise terms, and learning alpha deduced from Lambda.


import sys
import torch
import torch.optim as optim

sys.path.append("..")
from network.OCTOptimization import *
from network.QuadraticIPMOpt import *
from network.OCTAugmentation import *

sys.path.append(".")
from SurfaceSubnet import SurfaceSubnet
from ThicknessSubnet_M2 import ThicknessSubnet_M2
from LambdaModule_B import LambdaModule_B

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader


import os


class SoftSeparationNet_C(BasicModel):
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
        surfaceNetPath = self.hps.surfaceUpdatePath
        if len(os.listdir(surfaceNetPath)) == 0:
            surfaceNetPath = None  # network will load from default directory
        self.m_surfaceSubnet.m_netMgr.loadNet(surfaceMode, netPath=surfaceNetPath) # loadNet will load saved learning rate

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
        thicknessNetPath = self.hps.surfaceUpdatePath
        if len(os.listdir(thicknessNetPath)) == 0:
            thicknessNetPath = None  # network will load from default directory
        self.m_thicknessSubnet.m_netMgr.loadNet(thicknessMode, netPath=thicknessNetPath) # loadNet will load saved learning rate
        
        # lambda Module
        self.m_lDevice = eval(self.hps.lambdaModuleDevice)
        self.m_lambdaModule = eval(self.hps.lambdaModule)(self.m_surfaceSubnet.hps.startFilters+self.m_thicknessSubnet.hps.startFilters,\
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
        self.m_lambdaModule.setNetMgr(NetMgr(self.m_lambdaModule, self.hps.netPath, self.m_lDevice))
        lambdaNetPath = self.hps.surfaceUpdatePath
        if len(os.listdir(lambdaNetPath)) == 0:
            lambdaNetPath = None  # network will load from default directory
        self.m_lambdaModule.m_netMgr.loadNet(lambdaMode,netPath=lambdaNetPath)

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
        self.m_C[0, N - 1, N - 2] = 1.0
        for i in range(1, N - 1):
            self.m_C[0, i, i - 1] = 0.5
            self.m_C[0, i, i] = -0.5

        # define the 5-point center moving average smooth matrix
        W = hps.inputWidth
        self.m_smoothM = torch.zeros((1, W, W), dtype=torch.float32, device=hps.device,
                                     requires_grad=False)  # 5-point smooth matrix
        # 0th column and W-1 column
        self.m_smoothM[0, 0, 0] = 1.0 / 2
        self.m_smoothM[0, 1, 0] = 1.0 / 2
        self.m_smoothM[0, W - 2, W - 1] = 1.0 / 2
        self.m_smoothM[0, W - 1, W - 1] = 1.0 / 2
        # 1th column and W-2 column
        self.m_smoothM[0, 0, 1] = 1.0 / 3
        self.m_smoothM[0, 1, 1] = 1.0 / 3
        self.m_smoothM[0, 2, 1] = 1.0 / 3
        self.m_smoothM[0, W - 3, W - 2] = 1.0 / 3
        self.m_smoothM[0, W - 2, W - 2] = 1.0 / 3
        self.m_smoothM[0, W - 1, W - 2] = 1.0 / 3
        # columns from 2 to W-2
        for i in range(2, W - 2):
            self.m_smoothM[0, i - 2, i] = 1.0 / 5
            self.m_smoothM[0, i - 1, i] = 1.0 / 5
            self.m_smoothM[0, i, i] = 1.0 / 5
            self.m_smoothM[0, i + 1, i] = 1.0 / 5
            self.m_smoothM[0, i + 2, i] = 1.0 / 5

        self.m_A = torch.zeros((1, N-1, N), dtype=torch.float32, device=self.m_lDevice, requires_grad=False)
        for i in range(0, N - 1):
            self.m_A[0, i, i] = 1.0
            self.m_A[0, i, i+1] = -1.0

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

        # define some matrix for model 3
        self.m_bigA = torch.zeros((1, (N-1)*W, N*W), dtype=torch.float32, device=self.m_lDevice, requires_grad=False)
        for i in range(0, (N - 1)*W):
            self.m_bigA[0, i, i] = 1.0
            self.m_bigA[0, i, i+W] = -1.0

        self.m_bigD = torch.zeros((1, (N-1)*W, (N-1)*W), dtype=torch.float32, device=self.m_lDevice, requires_grad=False)
        Dt = self.m_D.transpose(-1,-2) # size: 1xWxW
        for i in range(0,N-1):
            self.m_bigD[0,i*W:(i+1)*W,i*W:(i+1)*W] = Dt


    def getSubnetModes(self):
        if self.hps.status == "trainLambda":
            surfaceMode = "test"
            thicknessMode = "test"
            lambdaMode = "train"
            self.generateSubnetUpdatePaths()
        elif self.hps.status == "fineTune":
            surfaceMode = "train"
            thicknessMode = "train"
            lambdaMode = "train"
            self.generateSubnetUpdatePaths()
        else:
            surfaceMode = "test"
            thicknessMode = "test"
            lambdaMode = "test"

        return surfaceMode, thicknessMode, lambdaMode


    def forward(self, images, imageYX, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        Mu, Sigma2, surfaceLoss, surfaceX = self.m_surfaceSubnet.forward(images.to(self.m_sDevice),
                                     gaussianGTs=gaussianGTs.to(self.m_sDevice),
                                     GTs=GTs.to(self.m_sDevice))

        # input channels: raw+Y+X
        R, thicknessLoss, thinknessX = self.m_thicknessSubnet.forward(imageYX.to(self.m_rDevice), gaussianGTs=None,GTs=None, layerGTs=layerGTs.to(self.m_rDevice),
                                                riftGTs= riftGTs.to(self.m_rDevice))

        # detach from surfaceSubnet and thicknessSubnet
        if self.hps.status == "trainLambda": # do not clone(), otherwise memory is huge.
            surfaceX = surfaceX.detach().to(self.m_lDevice)
            thinknessX = thinknessX.detach().to(self.m_lDevice)
            R = R.detach().to(self.m_lDevice)
            Mu = Mu.detach().to(self.m_lDevice)
            Sigma2 = Sigma2.detach().to(self.m_lDevice)  # size: nBxNxW

        # Lambda return backward propagation
        X = torch.cat((surfaceX, thinknessX), dim=1)
        Lambda = self.m_lambdaModule.forward(X)  # size: nBxNxW

        nB,nC,H,W = X.shape
        N = self.m_surfaceSubnet.hps.numSurfaces
        B = self.m_B.expand(nB, N, N)
        C = self.m_C.expand(nB, N, N - 1)
        M = self.m_smoothM.expand(nB, W, W)
        bigA = self.m_bigA.expand(nB,(N-1)*W, N*W)
        bigD = self.m_bigD.expand(nB,(N-1)*W, (N-1)*W)
        diagQ = torch.diag_embed(Sigma2.view(nB,-1),offset=0) # size: nBxNWxNW

        Alpha = 1.0- (Lambda[:,0:N-1,:] + Lambda[:,1:N,:])/2.0 # size: nBxN-1xW
        diagAlpha = torch.diag_embed(Alpha.view(nB,-1),offset=0) # size: nBx(N-1)Wx(N-1)W

        bmm = torch.bmm  # for concise notation

        S0 = bmm(Lambda*Mu+(1.0-Lambda)*(bmm(B, Mu)+bmm(C,R)), M) # size:nBxNxW
        vS0 = S0.view(nB,N*W,1)
        vR  = R.view(nB,(N-1)*W, 1)

        # intermediate variable Z with size: nBxNWx(N-1)W
        Z = bmm(bigA.transpose(-1,-2),bmm(bigD.transpose(-1,-2),bmm(diagAlpha,bigD)))
        # soft separation optimization model 3
        vS = bmm( torch.inverse(diagQ+bmm(Z,bigA)),(bmm(diagQ,vS0)-bmm(Z,vR)) ) #size: nBxNWx1
        S = vS.view(nB,N,W)

        for i in range(1, N):  #ReLU
            S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :], S[:, i - 1, :], S[:, i, :])

        # compute loss
        G = GTs.to(self.m_lDevice)
        lossFunc = torch.nn.SmoothL1Loss()  # L1 loss to avoid outlier exploding gradient.
        lambdaLoss = lossFunc(S,G)

        return S, surfaceLoss, thicknessLoss, lambdaLoss

    def zero_grad(self):
        if None != self.m_surfaceSubnet.m_optimizer:
            self.m_surfaceSubnet.m_optimizer.zero_grad()
        if None != self.m_thicknessSubnet.m_optimizer:
            self.m_thicknessSubnet.m_optimizer.zero_grad()
        if (None != self.m_lambdaModule) and (None != self.m_lambdaModule.m_optimizer):
            self.m_lambdaModule.m_optimizer.zero_grad()

    def backward(self, surfaceLoss, thickLoss, lambdaLoss):
       if self.hps.status == "trainLambda":
            lambdaLoss.backward(gradient=torch.ones(lambdaLoss.shape).to(lambdaLoss.device))
       elif self.hps.status == "fineTune":
           loss = surfaceLoss + thickLoss + lambdaLoss
           loss.backward(gradient=torch.ones(loss.shape).to(loss.device))
       else:
           pass

    def optimizerStep(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaModule.m_optimizer.step()
        elif  self.hps.status == "fineTune":
            self.m_lambdaModule.m_optimizer.step()
            self.m_surfaceSubnet.m_optimizer.step()
            self.m_thicknessSubnet.m_optimizer.step()
        else:
            pass

    def lrSchedulerStep(self, surfaceLoss, thickLoss, lambdaLoss):
        if self.hps.status == "trainLambda":
            self.m_lambdaModule.m_lrScheduler.step(lambdaLoss)
        elif self.hps.status == "fineTune":
            self.m_lambdaModule.m_lrScheduler.step(lambdaLoss)
            self.m_surfaceSubnet.m_lrScheduler.step(surfaceLoss)
            self.m_thicknessSubnet.m_lrScheduler.step(thickLoss)
        else:
            pass

    def saveNet(self):
        if self.hps.status == "trainLambda":
            self.m_lambdaModule.m_netMgr.saveNet(netPath=self.hps.lambdaUpdatePath)
        elif self.hps.status == "fineTune":
            self.m_lambdaModule.m_netMgr.saveNet(netPath=self.hps.lambdaUpdatePath)
            self.m_surfaceSubnet.m_netMgr.saveNet(netPath=self.hps.surfaceUpdatePath)
            self.m_thicknessSubnet.m_netMgr.saveNet(netPath=self.hps.thickUpdatePath)
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

    def generateSubnetUpdatePaths(self):
        netPath = self.hps.netPath

        self.hps.surfaceUpdatePath = os.path.join(netPath, "surfaceSubnet")
        if not os.path.exists(self.hps.surfaceUpdatePath):
            os.makedirs(self.hps.surfaceUpdatePath)  # recursive dir creation

        self.hps.thickUpdatePath = os.path.join(netPath, "thickSubnet")
        if not os.path.exists(self.hps.thickUpdatePath):
            os.makedirs(self.hps.thickUpdatePath)  # recursive dir creation

        self.hps.lambdaUpdatePath = os.path.join(netPath, "lambdaSubnet")
        if not os.path.exists(self.hps.lambdaUpdatePath):
            os.makedirs(self.hps.lambdaUpdatePath)  # recursive dir creation