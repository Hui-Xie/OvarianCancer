# Learninng Lambda module
# train on all training data, and validation on validation data

import sys
import torch

sys.path.append("..")
from network.OCTOptimization import *
from network.OCTAugmentation import *

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import  logits2Prob
from framework.ConfigReader import ConfigReader

class LambdaModule_D(BasicModel):
    def __init__(self, C,N, H, W, hps=None):
        '''
        inputSize: Bx(SurfaceSubnetChannel+ ThicknessSubnetChannel)xHxW
        C: the number of input channels
        N: the number of surfaces
        H: image height
        W: image width
        outputSize: (B, N, W) in [0,1]
        '''
        super().__init__()

        # Lambda branch:
        self.m_lambdas0 = nn.Sequential(
            Conv2dBlock(C, C // 2),
        )

        self.m_lambdas1 = nn.Sequential(  # residual module.
            Conv2dBlock(C//2, C // 2),
            Conv2dBlock(C // 2, C // 2),
            Conv2dBlock(C // 2, C // 2),
        )

        self.m_lambdas2 = nn.Sequential(
            nn.Conv2d(C // 2, hps.lambdaOutputC, kernel_size=[1, 1], stride=[1, 1], padding=[0, 0], bias=False),
        )# output size:BxlambaOutputCxHxW

        '''
        There are 3 methods to reduce H to 1: 
        A   softmax  on dimension H, and then use softargmax;
        B   using mean after 1x1 conv without norm to reduce H to 1; it looks also diverge in thickness network.
        C   use [H,1] convolution: as [H,1] convolution has too much parameters, it is easy to lead diverge. 
        '''
        # use method A
        #  in column, a prob distribution [0,0.001, 0.002, ..., maxAlpha ] in H dimension.
        self.m_probDistr = torch.arange(0, hps.maxAlpha, step= hps.maxAlpha/H).view(1, 1, H, 1)

    def forward(self, inputs):
        # N is numSurfaces
        x = self.m_lambdas0(inputs)
        x = x + self.m_lambdas1(x)  # residual module
        x = self.m_lambdas2(x)     # output size: Bxhps.LambdaOutputCxHxW
        x = logits2Prob(x, dim=-2)  # size: Bxhps.LambdaOutputCxHxW in softmax prob
        probLoc = self.m_probDistr.to(device=x.device).expand(x.size())  # prob locations
        lambdas = torch.sum(x*probLoc, dim=-2, keepdim=False)
        return lambdas  # return lambdas  in (B,hps.LambdaOutputC,W) dimension
