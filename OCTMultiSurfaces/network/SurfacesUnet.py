# Surfaces Unet: support automatic input size.


import sys
import math
import torch

sys.path.append(".")
from OCTOptimization import *
from OCTPrimalDualIPM import *
from QuadraticIPMOpt import *
from OCTAugmentation import *

sys.path.append("../..")
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import  GeneralizedDiceLoss, MultiLayerCrossEntropyLoss, MultiSurfaceCrossEntropyLoss, SmoothSurfaceLoss, logits2Prob, WeightedDivLoss


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
    def __init__(self, hps=None):
        '''
        inputSize: inputChaneels*H*W
        outputSize: (Surface, H, W)
        :param numSurfaces:
        :param N: startFilters
        '''
        super().__init__()
        self.hps = hps
        N = self.hps.startFilters

        self.m_layerSizeList = computeLayerSizeUsingMaxPool2D(self.hps.inputHeight, self.hps.inputWidth, self.hps.nLayers)

        self.m_useSpectralNorm = False
        self.m_useLeakyReLU = True
        # downxPooling layer is responsible change size of feature map (by MaxPool) and number of filters.
        self.m_down0Pooling = nn.Sequential(
            Conv2dBlock(self.hps.inputChannels, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
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
        )# output size: BxNxHxW

        # maybe 3 branches:
        self.m_surfaces = nn.Sequential(
            Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            nn.Conv2d(N, self.hps.numSurfaces, kernel_size=1, stride=1, padding=0)  # conv 1*1
        )  # output size:numSurfaces*H*W

        if self.hps.useLayerDice:
            self.m_layers = nn.Sequential(
                Conv2dBlock(N, N, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=self.m_useLeakyReLU),
                nn.Conv2d(N, self.hps.numSurfaces + 1, kernel_size=1, stride=1, padding=0)  # conv 1*1
            )  # output size:(numSurfaces+1)*H*W

        #tensors need switch H and W dimension to feed into self.m_rifts
        #The output of self.m_rifts need to squeeze the final dimension
        if hasattr(self.hps, 'useRiftWidth') and self.hps.useRiftWidth:
            self.m_rifts= nn.Sequential(
                Conv2dBlock(N, self.hps.numSurfaces, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=self.m_useLeakyReLU, kernelSize=3, padding=3, dilation=3), # output size: BxNumSurfacesxWxH
                nn.Linear(self.hps.inputHeight, 1),
                nn.ReLU()   # RiftWidth >=0
                )  # output size:numSurfaces*W*1
            #todo: R and sigma2 in Optimization do not need gradient backward(detached.)

    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        # compute outputs
        device = inputs.device

        x0 = self.m_down0Pooling(inputs)
        x0 = self.m_down0(x0) + x0

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
        if self.hps.useLayerDice:
            xl = self.m_layers(x)  # xs means x_layers,   # output size: B*(numSurfaces+1)*H*W
        if hasattr(self.hps, 'useRiftWidth') and self.hps.useRiftWidth:
            # tensors need switch H and W dimension to feed into self.m_rifts
            # The output of self.m_rifts need to squeeze the final dimension
            R = self.m_rifts(x.transpose(dim0=-1,dim1=-2))  # size: B*NumSurface*W*1
            R = R.squeeze(dim=-1) # size: B*N*W

        B,N,H,W = xs.shape

        layerMu = None # referred surface mu computed by layer segmentation.
        layerConf = None
        surfaceProb = logits2Prob(xs, dim=-2)
        layerProb = None
        if self.hps.useLayerDice:
            layerProb = logits2Prob(xl, dim=1)

        layerWeight = None
        surfaceWeight = None
        _, C, _, _ = inputs.shape
        if C >= 4: # at least 3 gradient channels.
            imageGradMagnitude = inputs[:, C-1, :, :]  # image gradient magnitude is at final channel since July 23th, 2020
            if self.hps.useLayerCE:
                layerWeight = getLayerWeightFromImageGradient(imageGradMagnitude, GTs, N + 1)
            surfaceWeight = getSurfaceWeightFromImageGradient(imageGradMagnitude, N, gradWeight=self.hps.gradWeight)

        loss_layer = 0.0
        if self.hps.useLayerDice:
            generalizedDiceLoss = GeneralizedDiceLoss()
            loss_layer = generalizedDiceLoss(layerProb, layerGTs) if self.hps.existGTLabel else 0.0

            if self.hps.useLayerCE and self.hps.existGTLabel:
                multiLayerCE = MultiLayerCrossEntropyLoss(weight=layerWeight)
                loss_layer += multiLayerCE(layerProb, layerGTs)

            if self.hps.useReferSurfaceFromLayer:
                layerMu, layerConf = layerProb2SurfaceMu(layerProb)  # use layer segmentation to refer surface mu.

        # compute surface mu and variance
        mu, sigma2 = computeMuVariance(surfaceProb, layerMu=layerMu, layerConf=layerConf)

        loss_surface = 0.0
        loss_smooth = 0.0
        if self.hps.existGTLabel:
            if self.hps.useCEReplaceKLDiv:
                multiSufaceCE = MultiSurfaceCrossEntropyLoss(weight=surfaceWeight)
                loss_surface = multiSufaceCE(surfaceProb, GTs)  # CrossEntropy is a kind of KLDiv

            elif self.hps.useWeightedDivLoss:
                weightedDivLoss = WeightedDivLoss(weight=surfaceWeight ) # the input given is expected to contain log-probabilities
                if 0 == len(gaussianGTs):  # sigma ==0 case
                    gaussianGTs = batchGaussianizeLabels(GTs, sigma2, H)
                loss_surface = weightedDivLoss(nn.LogSoftmax(dim=2)(xs), gaussianGTs)

            else:
                klDivLoss = nn.KLDivLoss(reduction='batchmean').to(device)
                # the input given is expected to contain log-probabilities
                if 0 == len(gaussianGTs):  # sigma ==0 case
                    gaussianGTs = batchGaussianizeLabels(GTs, sigma2, H)
                loss_surface = klDivLoss(nn.LogSoftmax(dim=2)(xs), gaussianGTs)

            if self.hps.useSmoothSurface:
                smoothSurfaceLoss = SmoothSurfaceLoss(mseLossWeight=10.0)
                loss_smooth = smoothSurfaceLoss(mu, GTs)

        weightL1 = 10.0
        l1Loss = nn.SmoothL1Loss().to(device)

        # rift L1 loss
        loss_riftL1 = 0.0
        if self.hps.existGTLabel and hasattr(self.hps, 'useRiftWidth') and self.hps.useRiftWidth:
            loss_riftL1 = l1Loss(R,riftGTs)

        S = mu
        if self.hps.usePrimalDualIPM:
            if self.hps.hardSeparation:
                separationIPM = HardSeparationIPMModule()
                S = separationIPM(mu, sigma2)
            elif self.hps.softSeparation:
                separationIPM = SoftSeparationIPMModule()
                S = separationIPM(mu, sigma2, R)
            else:
                print("Error: model parameter errs.")
                assert(False)

        loss_surfaceL1 = 0.0
        if self.hps.existGTLabel:
            loss_surfaceL1 = l1Loss(S, GTs)

        loss = loss_layer + loss_surface + loss_smooth+ (loss_surfaceL1 +loss_riftL1)* weightL1

        if torch.isnan(loss.sum()): # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return S, loss  # return surfaceLocation S in (B,S,W) dimension and loss



