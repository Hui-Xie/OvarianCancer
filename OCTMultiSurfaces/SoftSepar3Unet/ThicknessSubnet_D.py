# ThicknessSubnet: RiftNet_2

import sys
import torch.nn as nn

sys.path.append("..")
from network.OCTAugmentation import batchGaussianizeLabels
from network.OCTOptimization import computeMuVariance

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import logits2Prob, SmoothThicknessLoss
from framework.ConfigReader import ConfigReader

'''
In deep learning, smoothNet tries to learn the distance of adjacent pixels along their surface direction, 
here the adjacent pixels  are in 3x3 neighborhoods where 3x3 convolution and gradient information 
along the surface  will be helpful to get some information. 

While thicknessSubNet tries to learn the distance of adjacent surfaces along column direction. 
here the adjacent surfaces are most not in 3x3 or 5x5 neighborhoods. 
In Tongren data, the average distance of adjacent surfaces is 9 pixels, and maximal distance is 49 pixels.  
Learning R requires network to learn 2 information intuitively : 
A . the next computing point of distance is the adjacent surface, instead of a middle or next region point;  
B. what is the distance to his next adjacent surface.  
In this context, a simple 3x3 convolution is not enough to help solve this problem.  
If Deep learning is learning in a similar mode of human brain thinking, 
learning S (surface position) is a basis for learning R(separation). 
'''

# use 1D [H,1] convolution at end of Unet.

class ThicknessSubnet_D(BasicModel):
    def __init__(self, hps=None):
        '''
        inputSize: BxinputChaneels*H*W
        outputSize: (B, N-1, H, W), where N is the number of surfaces.
        '''
        super().__init__()
        if isinstance(hps, str):
            hps = ConfigReader(hps)
        self.hps = hps
        C = self.hps.startFilters

        # input of Unet: BxinputChannelsxHxW
        self.m_downPoolings, self.m_downLayers, self.m_upSamples, self.m_upLayers = \
            constructUnet(self.hps.inputChannels, self.hps.inputHeight, self.hps.inputWidth, C, self.hps.nLayers)
        # output of Unet: BxCxHxW

        # thickness branch
        self.m_thicknesses = nn.Sequential(
            Conv2dBlock(C, C//2, useLeakyReLU=True, kernelSize=5, padding=2),
            nn.Conv2d(C // 2, hps.numSurfaces - 1, kernel_size=[hps.inputHeight, 1], stride=[1, 1], padding=[0, 0]),  # 1D conv [H,1]
            nn.ReLU(),  # reLU make assure thickness >=0
        )  # output size:BxNx1xW

    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        device = inputs.device
        # compute outputs
        skipxs = [None for _ in range(self.hps.nLayers)]  # skip link x
        dropoutLayer = [nn.Dropout2d(p=self.hps.dropoutRateUnet[i], inplace=True) for i in range(self.hps.nLayers)]

        # down path of Unet
        for i in range(self.hps.nLayers):
            if 0 == i:
                x = inputs
            else:
                x = skipxs[i - 1]
            x = self.m_downPoolings[i](x)
            skipxs[i] = self.m_downLayers[i](x) + x

        # up path of Unet
        for i in range(self.hps.nLayers - 1, -1, -1):
            if self.hps.nLayers - 1 == i:
                x = skipxs[i]
            else:
                x = x + skipxs[i]
            x = dropoutLayer[i](x)    # dropout only at expand path, which make ThicknessNetwork different with SurfaceNet.
            x = self.m_upLayers[i](x) + x
            x = self.m_upSamples[i](x)
        # output of Unet: BxCxHxW

        # N is numSurfaces
        xt = self.m_thicknesses(x)  # xs means x_thickess, # output size: B*N*1*W
        thickness = xt.squeeze(dim=-2)  # size: Bx(numSurface-1)xW


        R = thickness

        # use smoothLoss and L1loss for rift
        loss_riftL1 = 0.0
        loss_smooth = 0.0
        if self.hps.existGTLabel:
            if self.hps.useL1Loss:
                l1Loss = nn.SmoothL1Loss().to(device)
                loss_riftL1 = l1Loss(R,riftGTs)
            if self.hps.useSmoothThicknessLoss:
                smoothThicknessLoss = SmoothThicknessLoss(mseLossWeight=10.0)
                loss_smooth = smoothThicknessLoss(R, riftGTs)

        loss = loss_riftL1 + loss_smooth

        if torch.isnan(loss.sum()): # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return thickness, loss  # return rift R in (B,N-1,W) dimension and loss



