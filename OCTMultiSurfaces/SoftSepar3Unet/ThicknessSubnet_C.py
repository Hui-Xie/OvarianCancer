# ThicknessSubnet: RiftNet_2
# Add cumulative thickness loss

import sys
import torch.nn as nn

sys.path.append("..")
from network.OCTAugmentation import batchGaussianizeLabels
from network.OCTOptimization import computeMuVariance

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import logits2Prob, SmoothThicknessLoss, CumulativeThicknessLoss
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

class ThicknessSubnet(BasicModel):
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

        # Surface branch
        self.m_surfaces = nn.Sequential(
            Conv2dBlock(C, C//2, useLeakyReLU=True, kernelSize=5, padding=2),
            Conv2dBlock(C//2, C//2, useLeakyReLU=True, kernelSize=7, padding=3),  # different from surfaceNet
            nn.Conv2d(C//2, self.hps.numSurfaces, kernel_size=1, stride=1, padding=0)  # conv 1*1
        )  # output size:BxNxHxW

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
        xs = self.m_surfaces(x)  # xs means x_surfaces, # output size: B*N*H*W
        B, N, H, W = xs.shape

        surfaceProb = logits2Prob(xs, dim=-2)

        # compute surface mu and variance
        mu, sigma2 = computeMuVariance(surfaceProb)  # size: B,N W
        thickness = mu[:,1:,:]-mu[:,0:-1,:] # size: B,N-1,W
        R = thickness

        # use smoothLoss and L1loss for rift
        loss_riftL1 = 0.0
        loss_smooth = 0.0
        loss_cumsum = 0.0
        if self.hps.existGTLabel:
            if self.hps.useL1Loss:
                l1Loss = nn.SmoothL1Loss().to(device)
                loss_riftL1 = l1Loss(R,riftGTs)
            if self.hps.useSmoothThicknessLoss:
                smoothThicknessLoss = SmoothThicknessLoss(mseLossWeight=10.0)
                loss_smooth = smoothThicknessLoss(R, riftGTs)
            if self.hps.useCumulativeThicknessLoss:
                cumulativeThicknessLoss = CumulativeThicknessLoss()
                loss_cumsum = cumulativeThicknessLoss(R, riftGTs)
                # in some sense, cumulativeThicknessLoss augments MSE.

        loss = loss_riftL1 + loss_smooth + loss_cumsum

        if torch.isnan(loss.sum()): # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        zeroThickness = torch.zeros_like(thickness)
        thickness = torch.where(thickness < zeroThickness, zeroThickness, thickness)  # make sure thickness >=0

        return thickness, loss  # return rift R in (B,N-1,W) dimension and loss



