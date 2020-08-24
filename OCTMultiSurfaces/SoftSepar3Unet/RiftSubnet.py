# Rift  Subnet

import sys
import torch.nn as nn

sys.path.append("..")
from network.OCTAugmentation import batchGaussianizeLabels

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import logits2Prob, SmoothSurfaceLoss
from framework.ConfigReader import ConfigReader

'''
In deep learning, smoothNet tries to learn the distance of adjacent pixels along their surface direction, 
here the adjacent pixels  are in 3x3 neighborhoods where 3x3 convolution and gradient information 
along the surface  will be helpful to get some information. 

While sepNet tries to learn the distance of adjacent surfaces along column direction. 
here the adjacent surfaces are most not in 3x3 or 5x5 neighborhoods. 
In Tongren data, the average distance of adjacent surfaces is 9 pixels, and maximal distance is 49 pixels.  
Learning R requires network to learn 2 information intuitively : 
A . the next computing point of distance is the adjacent surface, instead of a middle or next region point;  
B. what is the distance to his next adjacent surface.  
In this context, a simple 3x3 convolution is not enough to help solve this problem.  
If Deep learning is learning in a similar mode of human brain thinking, 
learning S (surface position) is a basis for learning R(separation). 
'''

class RiftSubnet(BasicModel):
    def __init__(self, hps=None):
        '''
        inputSize: BxinputChaneels*H*W
        outputSize: (B, N-1, H, W)
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

        #output (numSurfaces-1) rifts.
        self.m_rifts= nn.Sequential(
            Conv2dBlock(C, C),

            # 20200822 added 2 conv blocks
            Conv2dBlock(C, C),
            Conv2dBlock(C, C),
            # 20200822 added 2 conv blocks

            nn.Conv2d(C, self.hps.numSurfaces-1, kernel_size=1, stride=1, padding=0)  # conv 1*1
            )  # output size:Bx(N-1)xHxW



    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        device = inputs.device
        # compute outputs
        skipxs = [None for _ in range(self.hps.nLayers)]  # skip link x

        if 0 != self.hps.imageVerticalShift:
            a = self.hps.imageVerticalShift
            inputsShift = torch.cat((inputs[:,:,a:,:], inputs[:,:,0:a,:]), dim=-2)  #BxCxHxW
            inputs = torch.cat((inputs, inputsShift),dim=1)   # Bx2CxHxW

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
            x = self.m_upLayers[i](x) + x
            x = self.m_upSamples[i](x)
        # output of Unet: BxCxHxW

        # N is numSurfaces
        xr = self.m_rifts(x)  # output size: Bx(N-1)xHxW
        B,N,H,W = xr.shape

        riftProb = logits2Prob(xr, dim=-2)
        if self.hps.useBetterRiftGaussian:
            R = argSoftmax(riftProb, rangeH=[-H / 2.0, H / 2.0]) * 4.0 * self.hps.maxRift / H  # size: Bx(N-1)xW
            R = nn.functional.relu(R)  # make sure R>=0
        else:
            R = argSoftmax(riftProb)*self.hps.maxRift/H  # size: Bx(N-1)xW

        # use smoothLoss and KLDivLoss for rift
        loss_riftL1 = 0.0
        loss_smooth = 0.0
        loss_div = 0.0
        if self.hps.existGTLabel:
            if self.hps.useL1Loss:
                l1Loss = nn.SmoothL1Loss().to(device)
                loss_riftL1 = l1Loss(R,riftGTs)

            if self.hps.useSmoothLoss:
                smoothRiftLoss = SmoothSurfaceLoss(mseLossWeight=10.0)
                loss_smooth = smoothRiftLoss(R, riftGTs)

            if self.hps.useKLDivLoss:
                klDivLoss = nn.KLDivLoss(reduction='batchmean').to(device)
                # the input given is expected to contain log-probabilities
                sigma2 = self.hps.sigma**2
                sigma2 = float(sigma2)*torch.ones_like(riftGTs)
                if self.hps.useBetterRiftGaussian:
                    gaussianRiftGTs = batchGaussianizeLabels(riftGTs*H/(4.0*self.hps.maxRift), sigma2, [-H/2.0, H/2.0])
                else:
                    gaussianRiftGTs = batchGaussianizeLabels(riftGTs*H/self.hps.maxRift, sigma2, H)  # very important conversion
                loss_div = klDivLoss(nn.LogSoftmax(dim=2)(xr), gaussianRiftGTs)

        loss = loss_riftL1 + loss_smooth + loss_div

        if torch.isnan(loss.sum()): # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return R, loss  # return rift R in (B,N-1,W) dimension and loss



