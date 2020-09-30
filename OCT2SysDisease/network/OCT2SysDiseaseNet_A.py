
'''
input: BxSlicePerEye, inputC, H, W
output: B, SlicePerEyexC  to classification head

'''
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.SE_BottleNeck import  V3Bottleneck

class OCT2SysDiseaseNet_A(nn.Module):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        inC = 16  # input Channel number for bottleneck
        self.m_inputConv = nn.Sequential(
            nn.Conv2d(hps.inputChannels, inC, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(inC),
            nn.ReLU6(inplace=True)
        )

        bottleneckConfig = [
            # kernel, expandSize, outputChannel,  SE,   NL,  stride,
            # modify all activations to ReLU6
            [3, 16, 16, False, 'RE', 1],
            [3, 64, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1],
            [5, 72, 40, True, 'RE', 2],
            [5, 120, 40, True, 'RE', 1],
            [5, 120, 40, True, 'RE', 1],
            [3, 240, 80, False, 'RE', 2],
            [3, 200, 80, False, 'RE', 1],
            [3, 184, 80, False, 'RE', 1],
            [3, 184, 80, False, 'RE', 1],
            [3, 480, 112, True, 'RE', 1],
            [3, 672, 112, True, 'RE', 1],
            [5, 672, 160, True, 'RE', 2],
            [5, 960, 160, True, 'RE', 1],
            [5, 960, 160, True, 'RE', 1],
        ]  # for MobileNet V3 big model
        self.m_bottleneckList = nn.ModuleList()
        for kernel, expandSize, outC, SE, NL, stride in bottleneckConfig:
            self.m_bottleneckList.append(
                V3Bottleneck(inC, outC, kernel=kernel, stride=stride, expandSize=expandSize, SE=SE, NL=NL))
            inC = outC

        self.hps.spaceD = hps.slicesPerEye
        if hps.bothEyes:
            self.hps.spaceD = hps.slicesPerEye *2

        outC = hps.classifierWidth[0]//self.hps.spaceD
        self.m_featureConv = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hps.outputChannels),
            nn.ReLU6(inplace=True)
        )

        # after m_conv2d_1, tensor need global mean on H and W dimension.
        # and shift the spaceD form B to channel.

        self.m_classifier = nn.Sequential(
            nn.Conv2d(hps.classifierWidth[0], hps.classifierWidth[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=hps.dropoutRate, inplace=True),
            nn.Conv2d(hps.classifierWidth[1], hps.classifierWidth[2], kernel_size=1, stride=1, padding=0, bias=False)
            )

    def forward(self, x):
        x = self.m_inputConv(x)
        for bottle in self.m_bottleneckList:
            x = bottle(x)
        x = self.m_featureConv(x)

        x = x.mean(dim=(2, 3), keepdim=True) # B,C,1,1
        B,C,_,_ = x.shape
        x = x.view(B//self.hps.spaceD, C*self.hps.spaceD, 1,1)

        x = self.m_classifier(x)
        return x

    def computeLoss(self, x, GTs=None):
        pass



