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
            Conv2dBlock(C//2, C//4),
            nn.Conv2d(C // 4, N, kernel_size=[H,1], stride=[1, 1], padding=[0, 0]),  # 2D conv [H,1]
            nn.Sigmoid(),  # Sigmoid makes lambda in [0,1]
        )  # output size:BxNxW



    def forward(self, inputs):
        # N is numSurfaces
        lambdas = self.m_lambdas(inputs)  # output size: BxNx1xW
        return lambdas.squeeze(dim=-2)  # return lambdas  in (B,N,W) dimension
