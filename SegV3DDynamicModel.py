import torch
import torch.nn as nn
import torch.nn.functional as F

from BasicModel import BasicModel
from ConvBlocks import *


#  3D model uspporting dynamic input size with batchsize =1

class SegV3DModel(BasicModel):
    def __init__(self, useConsistencyLoss=False):  # K is the final output classification number.
        super().__init__()

        # For dynamic input image size
        # at Oct 14th, 2019
        #
        self.m_useSpectralNorm = True
        self.m_useLeakyReLU = True
        self.m_useConsistencyLoss = useConsistencyLoss

        self.m_downBlocks=nn.ModuleList  # each block includes pooling, convs. Inside pooling, it includes a layer to change filter number.
        self.m_upBlocks=nn.ModuleList    # each block includes unpooling, cropping, convs. Inside unpooling, it includes a layer to change filter number.
        self.m_outputBlock = None

        nF = 64  # the number of filter at the 0 layer, with batchsize = 1
        preFilters = 1
        for layer in range(0, 7):
            curFilters = nF*(layer+1)
            if 0 == layer:
                pooling = nn.Sequential(
                          Conv3dBlock(1, curFilters, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                         )
            else:
                pooling = nn.Sequential(
                          nn.MaxPool3d(3, stride=2, padding=1),   # padding 1 is to make sure the after resample with scale 2, the output tensor >= same layer tensor in down path.
                          Conv3dBlock(preFilters, curFilters, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                          )
            convs = nn.Sequential(
                    Conv3dBlock(curFilters, curFilters, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                    Conv3dBlock(curFilters, curFilters, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                    Conv3dBlock(curFilters, curFilters, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                    )
            self.m_downBlocks.append((p))


        # downxPooling layer is responsible change shape of feature map and number of filters.
        self.m_down0Pooling = nn.Sequential(
            Conv3dBlock(1, 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )  # ouput size: 32*49*147*147
        self.m_down0 = nn.Sequential(
            Conv3dBlock(32, 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(32, 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(32, 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )  # ouput size: 32*49*147*147

        self.m_down1Pooling = nn.Sequential(
            nn.AvgPool3d(3, stride=2, padding=0),
            Conv3dBlock(32, 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )  # ouput size: 64*24*73*73
        self.m_down1 = nn.Sequential(
            Conv3dBlock(64, 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(64, 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(64, 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down2Pooling = nn.Sequential(
            nn.AvgPool3d(3, stride=2, padding=0),
            Conv3dBlock(64, 128, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )  # ouput size: 128*11*36*36
        self.m_down2 = nn.Sequential(
            Conv3dBlock(128, 128, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(128, 128, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(128, 128, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down3Pooling = nn.Sequential(
            nn.AvgPool3d(3, stride=2, padding=0),
            Conv3dBlock(128, 256, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )  # ouput size: 256*5*17*17
        self.m_down3 = nn.Sequential(
            Conv3dBlock(256, 256, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(256, 256, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(256, 256, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down4Pooling = nn.Sequential(
            nn.AvgPool3d(3, stride=2, padding=0),
            Conv3dBlock(256, 512, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )  # ouput size: 512*2*8*8
        self.m_down4 = nn.Sequential(
            Conv3dBlock(512, 512, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(512, 512, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(512, 512, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_down5Pooling1 = nn.Sequential(
            nn.AvgPool3d((2, 3, 3), stride=2, padding=0)
        )  # outputSize:512*1*3*3, followed squeeze
        self.m_down5Pooling2 = nn.Sequential(
            Conv2dBlock(512, 1024, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        ) # outputSize:1024*3*3
        self.m_down5 = nn.Sequential(
            Conv2dBlock(1024, 1024, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(1024, 1024, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv2dBlock(1024, 1024, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        # here needs unsqueeze at dim 1

        self.m_up5Pooling = nn.Sequential(
            nn.Upsample(size=(2, 8, 8), mode='trilinear'),  # ouput size: 1024*2*8*8
            Conv3dBlock(1024, 512, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                           useLeakyReLU=self.m_useLeakyReLU)
        )# ouput size: 512*2*8*8
        self.m_up5 = nn.Sequential(
            Conv3dBlock(512, 512, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(512, 512, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(512, 512, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up4Pooling = nn.Sequential(
            nn.Upsample(size=(5, 17, 17), mode='trilinear'),
            Conv3dBlock(512, 256, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
        )  # ouput size: 256*5*17*17
        self.m_up4 = nn.Sequential(
            Conv3dBlock(256, 256, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(256, 256, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(256, 256, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up3Pooling = nn.Sequential(
            nn.Upsample(size=(11, 36, 36), mode='trilinear'),
            Conv3dBlock(256, 128, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
        )  # ouput size: 128*11*36*36
        self.m_up3 = nn.Sequential(
            Conv3dBlock(128, 128, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(128, 128, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(128, 128, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up2Pooling = nn.Sequential(
            nn.Upsample(size=(24, 73, 73), mode='trilinear'),
            Conv3dBlock(128, 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
        )  # ouput size: 64*24*73*73
        self.m_up2 = nn.Sequential(
            Conv3dBlock(64, 64, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(64, 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(64, 64, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up1Pooling = nn.Sequential(
            nn.Upsample(size=(49, 147, 147), mode='trilinear'),
            Conv3dBlock(64, 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        )  # ouput size: 32*49*147*147
        self.m_up1 = nn.Sequential(
            Conv3dBlock(32, 32, convStride=1,
                        useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(32, 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU),
            Conv3dBlock(32, 32, convStride=1, useSpectralNorm=self.m_useSpectralNorm,
                        useLeakyReLU=self.m_useLeakyReLU)
        )

        self.m_up0 = nn.Sequential(
            nn.Conv3d(32, 2, kernel_size=1, stride=1, padding=0)   # conv 1*1*1
        )  # output size:2*49*147*147





    def forward(self, inputs, gts):
        # compute outputs
        x0 = self.m_down0Pooling(inputs)
        # x0 = self.m_down0(x0) + x0    # this residual link hurts dice performance.
        x0 = self.m_down0(x0)

        x1 = self.m_down1Pooling(x0)
        x1 = self.m_down1(x1) + x1

        x2 = self.m_down2Pooling(x1)
        x2 = self.m_down2(x2) + x2

        x3 = self.m_down3Pooling(x2)
        x3 = self.m_down3(x3) + x3

        x4 = self.m_down4Pooling(x3)
        x4 = self.m_down4(x4) + x4

        x5 = self.m_down5Pooling1(x4)
        x5 = torch.squeeze(x5, dim=2)
        x5 = self.m_down5Pooling2(x5)
        x5 = self.m_down5(x5) + x5  # bottom neck output

        x = torch.unsqueeze(x5, dim=2)
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


        '''
        if self.m_useConsistencyLoss:
            # featureTensor = x    #just use final feature before softmax
            featureTensor = torch.cat((inputs, x0, x), dim=1)   # use feature from the 2 ends of V model, plus input
        '''
        outputs = self.m_up0(x)


        if self.m_useConsistencyLoss:
           xMaxDim1, _ = torch.max(outputs, dim=1, keepdim=True)
           xMaxDim1 = xMaxDim1.expand_as(outputs)
           predictProb = F.softmax(outputs - xMaxDim1, 1)  # use xMaxDim1 is to avoid overflow.



        # compute loss (put loss here is to save main GPU memory)
        loss = torch.tensor(0.0).to(x.device)
        for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
            if weight == 0:
                continue
            lossFunc.to(x.device)
            loss += lossFunc(outputs, gts) * weight

        if self.m_useConsistencyLoss:
            loss += self.m_consistencyLoss(predictProb, gts)

        return outputs, loss
