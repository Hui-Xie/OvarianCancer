# Surfaces Unet: support automatic input size.


import sys
import math

sys.path.append(".")
from OCTOptimization import *
from OCTPrimalDualIPM import *

sys.path.append("../..")
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import  GeneralizedDiceLoss, MultiClassCrossEntropyLoss


def computeLayerSizeUsingMaxPool2D(H, W, nLayers, kernelSize=2, stride=2, padding=0, dilation=1):
    '''
    use MaxPool2D to change layer size.
    :param W:
    :param H:
    :param nLayers:
    :param kernelSize:
    :param stride:
    :param padding:
    :param dilation:
    :return: a list of layer size of tuples:
             list[i] indicate the outputsize of layer i
    '''
    layerSizeList= []
    layerSizeList.append((H,W))
    for i in range(1, nLayers):
        H =int(math.floor((H+2*padding-dilation*(kernelSize-1)-1)/stride+1))
        W =int(math.floor((W+2*padding-dilation*(kernelSize-1)-1)/stride+1))
        layerSizeList.append((H, W))
    return  layerSizeList

class SurfacesUnet(BasicModel):
    def __init__(self, inputHight, inputWidth, inputChannels=1, nLayers=7, numSurfaces=11, N=24):
        '''
        inputSize: inputChaneels*H*W
        outputSize: (Surface, H, W)
        :param numSurfaces:
        :param N: startFilters
        '''
        super().__init__()

        self.m_inputHeight = inputHight
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

        xs = self.m_surfaces(x)  # xs means x_surfaces, # output size: B*numSurfaces*128*1024
        xl = self.m_layers(x)  # xs means x_layers,   # output size: B*(numSurfaces+1)*128*1024

        '''
        useProxialIPM = self.getConfigParameter('useProxialIPM')
        useDynamicProgramming = self.getConfigParameter("useDynamicProgramming")
        usePrimalDualIPM = self.getConfigParameter("usePrimalDualIPM")

        '''
        useCEReplaceKLDiv = self.getConfigParameter("useCEReplaceKLDiv")

        generalizedDiceLoss = GeneralizedDiceLoss()
        loss_layerDice = generalizedDiceLoss(xl, layerGTs)

        if useCEReplaceKLDiv:
            CELoss = MultiClassCrossEntropyLoss()
            loss_surfaceKLDiv = CELoss(xs, GTs)
        else:
            klDivLoss = nn.KLDivLoss(reduction='batchmean').to(device)  # the input given is expected to contain log-probabilities
            loss_surfaceKLDiv = klDivLoss(nn.LogSoftmax(dim=-2)(xs), gaussianGTs)

        smoothL1Loss = nn.SmoothL1Loss().to(device)
        mu, sigma2 = computeMuVariance(nn.Softmax(dim=-2)(xs))
        B, N, W = mu.shape
        separationPrimalDualIPM = SeparationPrimalDualIPM(B, W, N, device=device)
        S = separationPrimalDualIPM(mu, sigma2)
        loss_L1 = smoothL1Loss(S, GTs)

        weightL1 = 10.0
        loss = loss_layerDice + loss_surfaceKLDiv + loss_L1 * weightL1

        return S, loss  # return surfaceLocation S in (B,S,W) dimension and loss



