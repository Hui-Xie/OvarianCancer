# Rift  Subnet

import sys

sys.path.append(".")
from OCTOptimization import *
from OCTAugmentation import *

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import logits2Prob
from framework.ConfigReader import ConfigReader

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
            nn.Conv2d(C, self.hps.numSurfaces-1, kernel_size=1, stride=1, padding=0)  # conv 1*1
            )  # output size:Bx(N-1)xHxW



    def forward(self, inputs, gaussianGTs=None, GTs=None, layerGTs=None, riftGTs=None):
        device = inputs.device
        # compute outputs
        skipxs = [None for _ in range(self.hps.nLayers)]  # skip link x

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
        R = argSoftmax(riftProb)*self.hps.maxRift/H  # size: Bx(N-1)xW

        l1Loss = nn.SmoothL1Loss().to(device)

        # rift L1 loss
        loss_riftL1 = 0.0
        if self.hps.existGTLabel:
            loss_riftL1 = l1Loss(R,riftGTs)

        loss = loss_riftL1

        if torch.isnan(loss.sum()): # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return R, loss  # return rift R in (B,N-1,W) dimension and loss



