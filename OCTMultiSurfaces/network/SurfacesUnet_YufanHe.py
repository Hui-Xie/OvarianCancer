# Surfaces Unet: support automatic input size.


import sys
import math

sys.path.append(".")
from OCTOptimization import *
from OCTPrimalDualIPM import *
from OCTAugmentation import *
import torch

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import  GeneralizedDiceLoss, MultiSurfaceCrossEntropyLoss, SmoothSurfaceLoss, logits2Prob, MultiLayerCrossEntropyLoss

class SurfacesUnet_YufanHe(BasicModel):
    ''''
    This network refer Yufan He paper: He,Y., Carass A., Liu, Yi., et al. (2019).:
    Fully Convolutional Boundary Regression for Retina OCT Segmentation.
    In: Medical Image Computing and Computer Assisted Intervention(MICCAI 2019).
    \doi{10.1007/978-3-030-32239-7\_14}
    '''
    def __init__(self, inputHeight, inputWidth, inputChannels=1, nLayers=7, numSurfaces=11, N=24):
        '''
        inputSize: inputChaneels*H*W
        outputSize: (Surface, H, W)
        :param numSurfaces:
        :param N: startFilters
        '''
        super().__init__()

        self.m_inputHeight = inputHeight
        self.m_inputWidth = inputWidth
        self.m_inputChannels = inputChannels
        self.m_nLayers = nLayers
        self.m_layerSizeList = computeLayerSizeUsingMaxPool2D(self.m_inputHeight, self.m_inputWidth, self.m_nLayers)

        self.m_useSpectralNorm = False
        self.m_useLeakyReLU = True
        # downxPooling layer is responsible change size of feature map (by MaxPool) and number of filters.
        self.m_down0Pooling = nn.Sequential(
            Conv2dBlock(self.m_inputChannels, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_down0 = nn.Sequential(
            Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down1Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(N, N * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_down1 = nn.Sequential(
            Conv2dBlock(N * 2, N * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 2, N * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 2, N * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down2Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(N * 2, N * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_down2 = nn.Sequential(
            Conv2dBlock(N * 4, N * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 4, N * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 4, N * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down3Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(N * 4, N * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_down3 = nn.Sequential(
            Conv2dBlock(N * 8, N * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 8, N * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 8, N * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down4Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(N * 8, N * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down4 = nn.Sequential(
            Conv2dBlock(N * 16, N * 16, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 16, N * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 16, N * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down5Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(N * 16, N * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down5 = nn.Sequential(
            Conv2dBlock(N * 32, N * 32, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 32, N * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 32, N * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down6Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(N * 32, N * 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down6 = nn.Sequential(
            Conv2dBlock(N * 64, N * 64, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 64, N * 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 64, N * 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        # this is the bottleNeck at bottom with size: self.m_layerSizeList[6]

        self.m_up6Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[5], mode='bilinear'),
            Conv2dBlock(N * 64, N * 32, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up6 = nn.Sequential(
            Conv2dBlock(N * 32, N * 32, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 32, N * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 32, N * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up5Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[4], mode='bilinear'),
            Conv2dBlock(N * 32, N * 16, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up5 = nn.Sequential(
            Conv2dBlock(N * 16, N * 16, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 16, N * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 16, N * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up4Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[3], mode='bilinear'),
            Conv2dBlock(N * 16, N * 8, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up4 = nn.Sequential(
            Conv2dBlock(N * 8, N * 8, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 8, N * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 8, N * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up3Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[2], mode='bilinear'),
            Conv2dBlock(N * 8, N * 4, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up3 = nn.Sequential(
            Conv2dBlock(N * 4, N * 4, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 4, N * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 4, N * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up2Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[1], mode='bilinear'),
            Conv2dBlock(N * 4, N * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up2 = nn.Sequential(
            Conv2dBlock(N * 2, N * 2, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 2, N * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N * 2, N * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up1Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[0], mode='bilinear'),
            Conv2dBlock(N * 2, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_up1 = nn.Sequential(
            Conv2dBlock(N, N, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        # 2 branches:
        self.m_surfaces = nn.Sequential(
            Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            nn.Conv2d(N, numSurfaces, kernel_size=1, stride=1, padding=0)  # conv 1*1
        )  # output size:numSurfaces*H*W

        self.m_layers = nn.Sequential(
            Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            nn.Conv2d(N, numSurfaces + 1, kernel_size=1, stride=1, padding=0)  # conv 1*1
        )  # output size:(numSurfaces+1)*H*W


    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None):
        # compute outputs
        device = inputs.device

        x0 = self.m_down0Pooling(inputs)
        x0 = self.m_down0(x0) + x0    # this residual link may hurts dice performance.
        x0 = self.m_down0(x0)

        x1 = self.m_down1Pooling(x0)
        x1 = self.m_down1(x1) + x1

        x2 = self.m_down2Pooling(x1)
        x2 = self.m_down2(x2) + x2

        x3 = self.m_down3Pooling(x2)
        x3 = self.m_down3(x3) + x3

        x4 = self.m_down4Pooling(x3)
        x4 = self.m_down4(x4) + x4

        x5 = self.m_down5Pooling(x4)
        x5 = self.m_down5(x5) + x5

        x6 = self.m_down6Pooling(x5)
        x6 = self.m_down6(x6) + x6

        x = self.m_up6Pooling(x6) + x5
        x = self.m_up6(x) + x

        x = self.m_up5Pooling(x) + x4
        x = self.m_up5(x) + x

        x = self.m_up4Pooling(x) + x3
        x = self.m_up4(x) + x

        x = self.m_up3Pooling(x) + x2
        x = self.m_up3(x) + x

        x = self.m_up2Pooling(x) + x1
        x = self.m_up2(x) + x

        x = self.m_up1Pooling(x) + x0
        x = self.m_up1(x) + x

        xs = self.m_surfaces(x)  # xs means x_surfaces, # output size: B*numSurfaces*H*W
        xl = self.m_layers(x)  # xs means x_layers,   # output size: B*(numSurfaces+1)*H*W


        B,N,H,W = xs.shape

        hps = self.hps

        layerMu = None # referred surface mu computed by layer segmentation.
        layerConf = None
        surfaceProb = logits2Prob(xs, dim=-2)
        layerProb = logits2Prob(xl, dim=1)

        if hps.useLayerDice:
            generalizedDiceLoss = GeneralizedDiceLoss()
            loss_layer = generalizedDiceLoss(layerProb, layerGTs)
            # layerMu, layerConf = layerProb2SurfaceMu(layerProb)  # use layer segmentation to refer surface mu.

            # add layer CE loss
            layerWeight = getLayerWeightFromImages(inputs, GTs, N+1)
            multiLayerCE = MultiLayerCrossEntropyLoss(weight=layerWeight)
            loss_layer += multiLayerCE(layerProb, layerGTs)

        else:
            loss_layerDice = 0.0

        mu, sigma2 = computeMuVariance(surfaceProb, layerMu=layerMu, layerConf=layerConf)

        multiSurfaceCE = MultiSurfaceCrossEntropyLoss()
        loss_surface = multiSurfaceCE(surfaceProb, GTs)  # CrossEntropy is a kind of KLDiv


        #ReLU to guarantee layer order
        S = mu.clone()
        for i in range(1,N):
            S[:,i,:] = torch.where(S[:, i,:]< S[:,i-1,:], S[:,i-1,:], S[:,i,:])

        l1Loss = nn.SmoothL1Loss().to(device)
        loss_L1 = l1Loss(S, GTs)

        loss = loss_layer + loss_surface + loss_L1

        return S, loss  # return surfaceLocation S in (B,N,W) dimension and loss



