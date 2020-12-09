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
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import  GeneralizedDiceLoss, MultiLayerCrossEntropyLoss, MultiSurfaceCrossEntropyLoss, SmoothSurfaceLoss, logits2Prob, WeightedDivLoss


class SurfacesUnet(BasicModel):
    def __init__(self, hps=None):
        '''
        inputSize: inputChaneels*H*W
        outputSize: (Surface, H, W)
        :param numSurfaces:
        :param C: startFilters
        '''
        super().__init__()
        self.hps = hps
        C = self.hps.startFilters

        self.m_layerSizeList = computeLayerSizeUsingMaxPool2D(self.hps.inputHeight, self.hps.inputWidth, self.hps.nLayers)

        self.m_useSpectralNorm = False
        self.m_useLeakyReLU = True
        # downxPooling layer is responsible change size of feature map (by MaxPool) and number of filters.
        self.m_down0Pooling = nn.Sequential(
            Conv2dBlock(self.hps.inputChannels, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_down0 = nn.Sequential(
            Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down1Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(C, C * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_down1 = nn.Sequential(
            Conv2dBlock(C * 2, C * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 2, C * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 2, C * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down2Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(C * 2, C * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_down2 = nn.Sequential(
            Conv2dBlock(C * 4, C * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 4, C * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 4, C * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down3Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(C * 4, C * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_down3 = nn.Sequential(
            Conv2dBlock(C * 8, C * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 8, C * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 8, C * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down4Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(C * 8, C * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down4 = nn.Sequential(
            Conv2dBlock(C * 16, C * 16, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 16, C * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 16, C * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down5Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(C * 16, C * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down5 = nn.Sequential(
            Conv2dBlock(C * 32, C * 32, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 32, C * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 32, C * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down6Pooling = nn.Sequential(
            nn.MaxPool2d(2, stride=2, padding=0),
            Conv2dBlock(C * 32, C * 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down6 = nn.Sequential(
            Conv2dBlock(C * 64, C * 64, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 64, C * 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 64, C * 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        # this is the bottleNeck at bottom with size: self.m_layerSizeList[6]

        self.m_up6Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[5], mode='bilinear'),
            Conv2dBlock(C * 64, C * 32, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up6 = nn.Sequential(
            Conv2dBlock(C * 32, C * 32, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 32, C * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 32, C * 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up5Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[4], mode='bilinear'),
            Conv2dBlock(C * 32, C * 16, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up5 = nn.Sequential(
            Conv2dBlock(C * 16, C * 16, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 16, C * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 16, C * 16, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up4Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[3], mode='bilinear'),
            Conv2dBlock(C * 16, C * 8, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up4 = nn.Sequential(
            Conv2dBlock(C * 8, C * 8, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 8, C * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 8, C * 8, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up3Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[2], mode='bilinear'),
            Conv2dBlock(C * 8, C * 4, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up3 = nn.Sequential(
            Conv2dBlock(C * 4, C * 4, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 4, C * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 4, C * 4, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up2Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[1], mode='bilinear'),
            Conv2dBlock(C * 4, C * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
        )
        self.m_up2 = nn.Sequential(
            Conv2dBlock(C * 2, C * 2, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 2, C * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C * 2, C * 2, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up1Pooling = nn.Sequential(
            nn.Upsample(size=self.m_layerSizeList[0], mode='bilinear'),
            Conv2dBlock(C * 2, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )
        self.m_up1 = nn.Sequential(
            Conv2dBlock(C, C, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )# output size: BxCxHxW

        # 3 branches:
        self.m_surfaces = nn.Sequential(
            Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            nn.Conv2d(C, self.hps.numSurfaces, kernel_size=1, stride=1, padding=0)  # conv 1*1
        )  # output size:BxNxHxW

        if self.hps.useLayerDice:
            self.m_layers = nn.Sequential(
                Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=self.m_useLeakyReLU),
                nn.Conv2d(C, self.hps.numSurfaces + 1, kernel_size=1, stride=1, padding=0)  # conv 1*1
            )  # output size:(numSurfaces+1)*H*W

        #tensors need switch H and W dimension to feed into self.m_rifts
        #The output of self.m_rifts need to squeeze the final dimension
        #output (numSurfaces-1) rifts.
        if hasattr(self.hps, 'useRiftInPretrain'):
            self.m_rifts= nn.Sequential(
                Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=self.m_useLeakyReLU),
                Conv2dBlock(C, (self.hps.numSurfaces-1), convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=self.m_useLeakyReLU, kernelSize=3, padding=3, dilation=3), # output size: Bx(NumSurfaces-1)xWxH
                nn.Linear(self.hps.inputHeight, 1),
                nn.ReLU()   # RiftWidth >=0
                )  # output size:Bx(numSurfaces-1)*W*1

        # mu and sigma need to unsqueeze dim=1 to feed in this layer
        # and output of this layer needs squeeze dim=1
        if hasattr(self.hps, 'useCalibrate') and self.hps.useCalibrate:
            self.m_calibrate = nn.Sequential(
                Conv2dBlock(1, C, convStride=1,
                            useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=self.m_useLeakyReLU),
                Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=self.m_useLeakyReLU),
                nn.Conv2d(C, 1, kernel_size=1, stride=1, padding=0)  # output:Bx1x(N-1)xW
                )

        # learningPairWise weight module
        # input to this module: superpose mu, sigma, and r in feature channels, Bx3x(N-1)xW
        # output of this module: Bx1x(N-1)xW, need squeeze for further use.
        if hasattr(self.hps, 'useLearningPairWeight') and self.hps.useLearningPairWeight:
            self.m_learnPairWeight = nn.Sequential(
                Conv2dBlock(3, C, convStride=1,
                            useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                Conv2dBlock(C, C, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=self.m_useLeakyReLU),
                Conv2dBlock(C, 1, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                            useLeakyReLU=False, normAffine=True)
            ) # output:Bx1x(N-1)xW

    def inPretrain(self):
        status = self.getStatus()
        if status == "training" or status == "validation":
            if self.m_epoch >= self.hps.epochsPretrain:  # for unary item pretrain
                return False
            else:
                return True
        elif status == "test":
            if self.m_runParametersDict['epoch'] >= self.hps.epochsPretrain:
                return False
            else:
                return True
        else:
            print(f"wrong status: {status}")
            assert False

    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        # compute outputs
        e = 1e-8
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

        # N is numSurfaces
        xs = self.m_surfaces(x)  # xs means x_surfaces, # output size: B*numSurfaces*H*W
        if self.hps.useLayerDice:
            xl = self.m_layers(x)  # xs means x_layers,   # output size: B*(numSurfaces+1)*H*W

        R = None
        if self.hps.useRiftInPretrain or (not self.inPretrain()):
            # tensors need switch H and W dimension to feed into self.m_rifts
            # The output of self.m_rifts need to squeeze the final dimension
            if self.hps.gradientRiftConvGoBack:
                xClone = x
            else:
                xClone = x.clone().detach()  # the gradient of rift module do not go back to Unet.
            R = self.m_rifts(xClone.transpose(dim0=-1,dim1=-2))  # size: B*(NumSurface-1)*W*1
            R = R.squeeze(dim=-1) # size: B*(N-1)*W

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
        mu, sigma2 = computeMuVariance(surfaceProb, layerMu=layerMu, layerConf=layerConf)  # size: B,N W

        if self.hps.useRiftInPretrain or (not self.inPretrain()):
            assert ((self.hps.useCalibrate and self.hps.useMergeMuRift and self.hps.useLearningPairWeight) == False)
            # todo: R and sigma2 need detach
            if self.hps.useCalibrate:
                R_sigma2 = R * sigma2[:,1:,:]  # size: Bx(N-1)xW
                R_sigma2 = R_sigma2.unsqueeze(dim=1) # outputsize: Bx1x(N-1)xW
                mu = torch.cat((mu[:,0,:].unsqueeze(dim=1), mu[:,1:,:] + self.m_calibrate(R_sigma2).squeeze(dim=1)), dim=1) # size: BxNxW
            if self.hps.useMergeMuRift:
                mu_R = mu[:,0:-1,:]+R #size: Bx(N-1)xW
                sigma2_i = 1.0/(sigma2[:,1:,:]+e) # reciprocal, size: Bx(N-1)xW
                sigma2_i_1 = 1.0/(sigma2[:,0:-1,:]+e) # reciprocal, size: Bx(N-1)xW
                mu = torch.cat((mu[:,0,:].unsqueeze(dim=1), (sigma2_i*mu[:,1:,:]+sigma2_i_1*mu_R)/(sigma2_i+ sigma2_i_1)), dim=1) # size: BxNxW

            pairWeight = None
            if self.hps.useLearningPairWeight:
                # the gradient of mu,sigma2, and R do not go back
                mu_ = mu[:,1:,:].clone().detach().unsqueeze(dim=1) # size: Bx1x(N-1)xW
                sigma2_ = sigma2[:,1:,:].clone().detach().unsqueeze(dim=1) # size: Bx1x(N-1)xW
                R_ = R.clone().detach().unsqueeze(dim=1) # size: Bx1x(N-1)xW
                muSigma2R = torch.cat((mu_,sigma2_,R_), dim=1)  # size: Bx3x(N-1)xW
                pairWeight = self.m_learnPairWeight(muSigma2R).squeeze(dim=1)   # size: Bx(N-1)xW
                #  Clamp on the learning lambda into range [0.1, 0.9], to avoid it is zero or  too big;
                #  This will solve the problem of the high condition number of matrix H;(Higher condition number, more close to singular.)
                pairWeight = torch.clamp(pairWeight, min=0.1, max=0.9)


        loss_surface = 0.0
        loss_smooth = 0.0
        if self.hps.existGTLabel:
            if self.hps.useCEReplaceKLDiv:
                multiSufaceCE = MultiSurfaceCrossEntropyLoss(weight=surfaceWeight)
                loss_surface = multiSufaceCE(surfaceProb, GTs)  # CrossEntropy is a kind of KLDiv

            elif self.hps.useWeightedDivLoss:
                weightedDivLoss = WeightedDivLoss(weight=surfaceWeight ) # the input given is expected to contain log-probabilities
                if 0 == len(gaussianGTs):  # sigma ==0 case, dynamic gausssian
                    gaussianGTs = batchGaussianizeLabels(GTs, sigma2, H)
                loss_surface = weightedDivLoss(nn.LogSoftmax(dim=2)(xs), gaussianGTs)

            else:
                klDivLoss = nn.KLDivLoss(reduction='batchmean').to(device)
                # the input given is expected to contain log-probabilities
                if 0 == len(gaussianGTs):  # sigma ==0 case, dynamic gausssian
                    gaussianGTs = batchGaussianizeLabels(GTs, sigma2, H)
                loss_surface = klDivLoss(nn.LogSoftmax(dim=2)(xs), gaussianGTs)

            if self.hps.useSmoothSurfaceLoss:
                smoothSurfaceLoss = SmoothSurfaceLoss(mseLossWeight=10.0)
                loss_smooth = smoothSurfaceLoss(mu, GTs)

        weightL1 = 10.0
        l1Loss = nn.SmoothL1Loss().to(device)

        # rift L1 loss
        loss_riftL1 = 0.0
        if self.hps.existGTLabel and (self.hps.useRiftInPretrain or (not self.inPretrain())):
            if self.hps.smoothRbeforeLoss:
                RSmooth = smoothCMA_Batch(R, self.hps.smoothHalfWidth, self.hps.smoothPadddingMode)
                loss_riftL1 = l1Loss(RSmooth, riftGTs)
            else:
                loss_riftL1 = l1Loss(R,riftGTs)

        if self.inPretrain() and self.hps.useReLUInPretrain:
            # ReLU to guarantee layer order not to cross each other
            S = mu.clone()
            for i in range(1, N):
                S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :], S[:, i - 1, :], S[:, i, :])
        else:
            separationIPM = SoftSeparationIPMModule()
            if self.hps.softSeparation:
                R_detach = R.clone().detach()
                S = separationIPM(mu, sigma2, R=R_detach, fixedPairWeight=self.hps.fixedPairWeight,
                                  learningPairWeight=pairWeight)
            else:
                if 2 == self.hps.hardSeparation:
                    S = separationIPM(mu, sigma2)
                elif 1 == self.hpa.hardSeparation:
                    S = mu.clone()
                    for i in range(1, N):
                        S[:, i, :] = torch.where(S[:, i, :] < S[:, i - 1, :], S[:, i - 1, :], S[:, i, :])
                else:  # No ReLU
                    S = mu.clone()


        loss_surfaceL1 = 0.0
        if self.hps.existGTLabel:
            loss_surfaceL1 = l1Loss(S, GTs)

        loss = loss_layer + loss_surface + loss_smooth+ (loss_surfaceL1 +loss_riftL1)* weightL1

        if torch.isnan(loss.sum()): # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        if self.hps.debug and (self.hps.useRiftInPretrain or (not self.inPretrain())):
            return S, loss, R
        else:
            return S, loss  # return surfaceLocation S in (B,S,W) dimension and loss



