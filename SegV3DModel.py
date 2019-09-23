import torch
import torch.nn as nn
import torch.nn.functional as F

from BasicModel import BasicModel
from ConvBlocks import *
#  3D model

class SegV3DModel (BasicModel):
    def __init__(self):   # K is the final output classification number.
        super().__init__()
        # For input image size: 51*171*171 (zyx in nrrd format)
        # at Sep 14, 2019,
        # log:
        #
        # result:
        #
        self.m_useSpectralNorm = True
        self.m_useLeakyReLU = True
        self.m_down0 = nn.Sequential(
            Conv3dBlock(1, 32, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(32, 32, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(32, 32, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 32*51*171*171
        self.m_down1 = nn.Sequential(
            Conv3dBlock(32, 64, poolingLayer=nn.AvgPool3d(3, stride=2, padding=0), convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(64, 64, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(64, 64, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 64*25*85*85
        self.m_down2 = nn.Sequential(
            Conv3dBlock(64, 128, poolingLayer=nn.AvgPool3d(3, stride=2, padding=0), convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(128, 128, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(128, 128, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 128*12*42*42

        self.m_down3 = nn.Sequential(
            Conv3dBlock(128, 256, poolingLayer=nn.AvgPool3d(3, stride=2, padding=0), convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(256, 256, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(256, 256, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 256*5*20*20
        self.m_down4 = nn.Sequential(
            Conv3dBlock(256, 512, poolingLayer=nn.AvgPool3d(3, stride=2, padding=0), convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(512, 512, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(512, 512, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 512*2*9*9

        self.m_up4 = nn.Sequential(
            Conv3dBlock(512, 256, poolingLayer=nn.Upsample(size=(5, 20, 20), mode='trilinear'), convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(256, 256, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(256, 256, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 256*5*20*20

        self.m_up3 = nn.Sequential(
            Conv3dBlock(256, 128, poolingLayer=nn.Upsample(size=(12, 42, 42), mode='trilinear'), convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(128, 128, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(128, 128, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 128*12*42*42
        self.m_up2 = nn.Sequential(
            Conv3dBlock(128, 64, poolingLayer=nn.Upsample(size=(25, 85, 85), mode='trilinear'), convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(64, 64, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(64, 64, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 64*25*85*85
        self.m_up1 = nn.Sequential(
            Conv3dBlock(64, 32, poolingLayer=nn.Upsample(size=(51, 171, 171), mode='trilinear'), convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(32, 32, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(32, 32, poolingLayer=None, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 32*51*171*171
        self.m_up0 = nn.Sequential(
            nn.Conv3d(32, 2, kernel_size=3,stride=1, padding=1)
           )  # output size:2*51*171*171


    def forward(self, x, gts):
            # compute outputs
            x0 = self.m_down0(x)
            x1 = self.m_down1(x0)
            x2 = self.m_down2(x1)
            x3 = self.m_down3(x2)
            x4 = self.m_down4(x3)

            x  = self.m_up4(x4) +x3
            x  = self.m_up3(x) + x2
            x  = self.m_up2(x) + x1
            x  = self.m_up1(x) + x0
            outputs  = self.m_up0(x)

            # compute loss (put loss here is to save main GPU memory)
            loss = torch.tensor(0.0).to(x.device)
            for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
                if weight == 0:
                    continue
                lossFunc.to(x.device)
                loss += lossFunc(outputs, gts) * weight

            return outputs, loss

