from BasicModel import BasicModel
from ResNeXtBlock import ResNeXtBlock
from SpatialTransformer import SpatialTransformer
from DeformConvBlock import DeformConvBlock
import torch.nn as nn
import torch
import torch.nn.functional as F

# ResNeXt V Net

class ResNeXtVNet(BasicModel):
    def forward(self, x):
        x0 = self.m_down0(x)
        x1 = self.m_down1(x0)
        x2 = self.m_down2(x1)
        x3 = self.m_down3(x2)
        x4 = self.m_down4(x3)
        x5 = self.m_down5(x4)
        x5 = x5.view(-1,512)
        x5 = F.normalize(x5,dim=1)
        x6 = self.m_bottomFC(x5)
        x6 = F.normalize(x6, dim=1)  # this is for output of latent vector
        latentV= x6
        x5 = x5.view(-1,512,1,1)+ x6.view(-1,512,1,1)
        x4 = x4 + self.m_up5(x5)
        x3 = x3 + self.m_up4(x4)
        x2 = x2 + self.m_up3(x3)
        x1 = x1 + self.m_up2(x2)
        x0 = x0 + self.m_up1(x1)
        out = self.m_up0(x0)
        return out

    def __init__(self):
        super().__init__()
        # For input image size: 231*251*251 (zyx)
        # at Aug 30, 2019,
        # log:  final FC layer has a 512 width.
        #
        # result:
        #
        self.m_useSpectralNorm = True
        self.m_useLeakyReLU = True
        self.m_down0 = nn.Sequential(
            ResNeXtBlock(231, 128, nGroups=33, poolingLayer=nn.AvgPool2d(3,stride=2, padding=0),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 128*125*125
        self.m_down1 = nn.Sequential(
            ResNeXtBlock(128, 128, nGroups=32, poolingLayer=nn.AvgPool2d(3, stride=2, padding=0),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # ouput size: 256*62*62
        self.m_down2 = nn.Sequential(
            ResNeXtBlock(256, 256, nGroups=32, poolingLayer=nn.AvgPool2d(3, stride=2, padding=0),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # output size: 512*30*30
        self.m_down3 = nn.Sequential(
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.AvgPool2d(3, stride=2, padding=0),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # output size: 512*14*14
        self.m_down4 = nn.Sequential(
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.AvgPool2d(3, stride=2, padding=0),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # output size: 512*6*6
        self.m_down5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=6, stride=6, padding=0, bias=True),  # Weighted Global pooling
            nn.ReLU(inplace=True) if not self.m_useLeakyReLU else nn.LeakyReLU(inplace=True)
            )  # output size: 512*1*1, it needs normalization.

        self.m_bottomFC = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True) if not self.m_useLeakyReLU else nn.LeakyReLU(inplace=True)
           )  # output size: 512*1, it needs normalization.

        self.m_up5 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=6, stide=6, padding=0, bias=True),
            nn.ReLU(inplace=True) if not self.m_useLeakyReLU else nn.LeakyReLU(inplace=True)
            ) # output size: 512*6*6, it needs normalization

        self.m_up4 = nn.Sequential(
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.Upsample(size=(14,14), mode='bilinear'),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # output size: 512*14*4
        self.m_up3 = nn.Sequential(
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.Upsample(size=(30, 30), mode='bilinear'),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # output size: 512*30*30
        self.m_up2 = nn.Sequential(
            ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.Upsample(size=(62, 62), mode='bilinear'),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(512, 256, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # output size: 256*62*62
        self.m_up1 = nn.Sequential(
            ResNeXtBlock(256, 256, nGroups=32, poolingLayer=nn.Upsample(size=(125, 125), mode='bilinear'),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(256, 128, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU)
            )  # output size: 128*125*125
        self.m_up0 = nn.Sequential(
            ResNeXtBlock(128, 128, nGroups=32, poolingLayer=nn.Upsample(size=(251, 251), mode='bilinear'),
                         useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm,
                         useLeakyReLU=self.m_useLeakyReLU),
            nn.Conv2d(128, 231, kernel_size=1, stride=1, padding=0, bias=True)
            )  # output size: 231*251*251

