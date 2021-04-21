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
        self.m_lambdas = nn.Sequential(
            Conv2dBlock(C, C//2),
            Conv2dBlock(C//2, C//4), # size: BxC//4xHxW
            nn.Conv2d(C // 4, N, kernel_size=[1,1], stride=[1, 1], padding=[0, 0],bias=False),
        )  # output size:BxNxHxW

        self.m_sizeFinalConvFilter = C//4

        '''
        There are 3 methods to reduce H to 1: 
        A   softmax  on dimension H, and then use softargmax;
        B   using mean after 1x1 conv without norm to reduce H to 1; it looks also diverge in thickness network.
        C   use [H,1] convolution: as [H,1] convolution has too much parameters, it is easy to lead diverge. 
        '''



    def forward(self, inputs):
        # N is numSurfaces
        x = self.m_lambdas(inputs)  # output size: BxNxHxW
        x = x/self.m_sizeFinalConvFilter   # average the output of conv to make the value small near zero.
        x = torch.sigmoid(x)   # first sigmoid, than mean to express voting mechanism.
        lambdas = torch.mean(x,dim=-2, keepdim=False) # outputsize: BxNxW
        return lambdas  # return lambdas  in (B,N,W) dimension
