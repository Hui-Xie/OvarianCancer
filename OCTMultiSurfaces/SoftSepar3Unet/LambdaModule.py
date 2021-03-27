# Learninng Lambda module
# train on all training data, and validation on validation data

import sys

sys.path.append("..")
from network.OCTOptimization import *
from network.OCTAugmentation import *

sys.path.append("../..")
from framework.NetTools import *
from framework.BasicModel import BasicModel
from framework.ConvBlocks import *
from framework.CustomizedLoss import  logits2Prob
from framework.ConfigReader import ConfigReader

class LambdaModule(BasicModel):
    def __init__(self, C,N, H, W):
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
        self.m_lambdas = nn.Sequential(
            Conv2dBlock(C, C//2),
            Conv2dBlock(C//2, C//4), # size: BxC//4xHxW
            nn.Conv2d(C // 4, N, kernel_size=[1,1], stride=[1, 1], padding=[0, 0]),
        )  # output size:BxNxHxW

        '''
        using mean after 1x1 conv without norm to reduce H to 1 is better than [H,1] convolution,
        as [H,1] convolution has too much parameters, it is easy to lead not converge. 
        '''



    def forward(self, inputs):
        # N is numSurfaces
        x = self.m_lambdas(inputs)  # output size: BxNxHxW
        x = torch.mean(x,dim=-2, keepdim=False) # outputsize: BxNxW
        lambdas = torch.sigmoid(x)   # output in [0,1] with size of BxNxW
        return lambdas  # return lambdas  in (B,N,W) dimension
