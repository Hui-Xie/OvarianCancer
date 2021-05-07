# merge the feature of surfaceUnet + thicknessUnet, directly deduce new surface location.


import sys
import torch
import torch.optim as optim

sys.path.append("..")
from network.OCTOptimization import *
from network.QuadraticIPMOpt import *
from network.OCTAugmentation import *

sys.path.append(".")
from SurfaceSubnet_Q import SurfaceSubnet_Q
from ThicknessSubnet_Q import ThicknessSubnet_Q

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader

import os


class MergeNet_Q(BasicModel):
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
        self.generateSubnetUpdatePaths()

        # surface Subnet
        self.m_sDevice = eval(self.hps.surfaceSubnetDevice)
        surfaceHps = ConfigReader(self.hps.surfaceSubnetYaml)
        surfaceHps.device = self.m_sDevice
        self.m_surfaceSubnet = eval(self.hps.surfaceSubnet)(hps=surfaceHps)
        self.m_surfaceSubnet.to(self.m_sDevice)
        self.m_surfaceSubnet.m_optimizer = None
        if "test" != surfaceMode:
            self.m_surfaceSubnet.setOptimizer(optim.Adam(self.m_surfaceSubnet.parameters(), lr=self.hps.surfaceSubnetLr, weight_decay=0))
            self.m_surfaceSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_surfaceSubnet.m_optimizer, \
                                            mode="min", factor=0.5, patience=10, min_lr=1e-8, threshold=0.02, threshold_mode='rel'))
        self.m_surfaceSubnet.setNetMgr(NetMgr(self.m_surfaceSubnet, self.m_surfaceSubnet.hps.netPath, self.m_sDevice))
        surfaceNetPath = self.hps.surfaceUpdatePath
        if len(os.listdir(surfaceNetPath)) == 0:
            surfaceNetPath = None  # network will load from default directory
        self.m_surfaceSubnet.m_netMgr.loadNet(surfaceMode, netPath=surfaceNetPath) # loadNet will load saved learning rate

        # thickness Subnet, where r means thickness
        self.m_rDevice = eval(self.hps.thicknessSubnetDevice)
        thicknessHps = ConfigReader(self.hps.thicknessSubnetYaml)
        thicknessHps.device = self.m_rDevice
        self.m_thicknessSubnet = eval(self.hps.thicknessSubnet)(hps=thicknessHps)
        self.m_thicknessSubnet.to(self.m_rDevice)
        self.m_thicknessSubnet.m_optimizer = None
        if "test" != thicknessMode:
            self.m_thicknessSubnet.setOptimizer(
                optim.Adam(self.m_thicknessSubnet.parameters(), lr=self.hps.thicknessSubnetLr, weight_decay=0))
            self.m_thicknessSubnet.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_thicknessSubnet.m_optimizer, \
                                                                                 mode="min", factor=0.5, patience=10,
                                                                                 min_lr=1e-8, threshold=0.02,
                                                                                 threshold_mode='rel'))
        self.m_thicknessSubnet.setNetMgr(NetMgr(self.m_thicknessSubnet, self.m_thicknessSubnet.hps.netPath, self.m_rDevice))
        thicknessNetPath = self.hps.thickUpdatePath
        if len(os.listdir(thicknessNetPath)) == 0:
            thicknessNetPath = None  # network will load from default directory
        self.m_thicknessSubnet.m_netMgr.loadNet(thicknessMode, netPath=thicknessNetPath) # loadNet will load saved learning rate
        
        # construct  merged segmentation part.
        assert (surfaceHps.segChannels == thicknessHps.segChannels)
        self.m_mergeSurfaces = nn.Sequential(
            Conv2dBlock(surfaceHps.startFilters+thicknessHps.startFilters, surfaceHps.segChannels),
            nn.Conv2d(surfaceHps.segChannels, self.hps.numSurfaces, kernel_size=1, stride=1, padding=0)  # conv 1*1
        )  # output size:BxNxHxW

        # lambda Module
        self.m_lDevice = eval(self.hps.lambdaModuleDevice)
        self.m_lambdaModule = eval(self.hps.lambdaModule)(self.m_surfaceSubnet.hps.startFilters+self.m_thicknessSubnet.hps.startFilters,\
                                                          self.m_surfaceSubnet.hps.segChannels,\
                                                          hps=hps)
        self.m_lambdaModule.to(self.m_lDevice)
        # =============== copy weight from pretrained surfaceSubnet and thicknessSubnet=============================
        self.m_lambdaModule.copyWeightFrom(self.m_surfaceSubnet.m_surfaces, self.m_thicknessSubnet.m_surfaces)
        # ==========================================================================================================
        self.m_lambdaModule.m_optimizer = None
        if "test" != lambdaMode:
            self.m_lambdaModule.setOptimizer(
                optim.Adam(self.m_lambdaModule.parameters(), lr=self.hps.lambdaModuleLr, weight_decay=0))
            self.m_lambdaModule.setLrScheduler(optim.lr_scheduler.ReduceLROnPlateau(self.m_lambdaModule.m_optimizer, \
                                                                              mode="min", factor=0.5, patience=20,
                                                                              min_lr=1e-8, threshold=0.02,
                                                                              threshold_mode='rel'))
        self.m_lambdaModule.setNetMgr(NetMgr(self.m_lambdaModule, self.hps.netPath, self.m_lDevice))
        lambdaNetPath = self.hps.lambdaUpdatePath
        if len(os.listdir(lambdaNetPath)) == 0:
            lambdaNetPath = None  # network will load from default directory
        self.m_lambdaModule.m_netMgr.loadNet(lambdaMode,netPath=lambdaNetPath)

        self.setSubnetsStatus(surfaceMode, thicknessMode, lambdaMode)

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

    def setSubnetsStatus(self,surfaceMode, thicknessMode, lambdaMode):
        self.m_surfaceSubnet.setStatus(surfaceMode)
        self.m_thicknessSubnet.setStatus(thicknessMode)
        self.m_lambdaModule.setStatus(lambdaMode)


    def forward(self, images, imageYX, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        Mu, Sigma2, surfaceLoss, surfaceX = self.m_surfaceSubnet.forward(images.to(self.m_sDevice),
                                     gaussianGTs=gaussianGTs.to(self.m_sDevice),
                                     GTs=GTs.to(self.m_sDevice), layerGTs=layerGTs.to(self.m_sDevice))

        # input channels: raw+Y+X
        R, thicknessLoss, thicknessX = self.m_thicknessSubnet.forward(imageYX.to(self.m_rDevice), gaussianGTs=None,GTs=None, layerGTs=layerGTs.to(self.m_rDevice),
                                                riftGTs= riftGTs.to(self.m_rDevice))

        # detach from surfaceSubnet and thicknessSubnet
        if self.hps.status == "trainLambda": # do not clone(), otherwise memory is huge.
            surfaceX = surfaceX.detach()
            thicknessX = thicknessX.detach()
            R = R.detach()
            Mu = Mu.detach()
            Sigma2 = Sigma2.detach()  # size: nBxNxW

        surfaceX = surfaceX.to(self.m_lDevice)
        thicknessX = thicknessX.to(self.m_lDevice)
        # R = R.to(self.m_lDevice)
        # Mu = Mu.to(self.m_lDevice)
        # Sigma2 = Sigma2.to(self.m_lDevice)
        #if not isinstance(surfaceLoss,float):
        #    surfaceLoss = surfaceLoss.to(self.m_lDevice)
        #if not isinstance(thicknessLoss, float):
        #    thicknessLoss = thicknessLoss.to(self.m_lDevice)

        # Lambda return backward propagation
        X = torch.cat((surfaceX, thicknessX), dim=1)
        # nB, nC, H, W = X.shape
        S, lambdaSigma2, lambdaLoss = self.m_lambdaModule.forward(X, GTs=GTs.to(self.m_lDevice))  # size: nBxNxW

        # free memory
        del surfaceX
        del thicknessX
        del X

        return S, 0, 0, lambdaLoss  # return S, surfaceLoss, thicknessLoss, lambdaLoss

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
            loss = surfaceLoss + thickLoss + lambdaLoss
            self.m_lambdaModule.m_lrScheduler.step(loss)
            self.m_surfaceSubnet.m_lrScheduler.step(loss)
            self.m_thicknessSubnet.m_lrScheduler.step(loss)
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