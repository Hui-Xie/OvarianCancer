
import torch.nn as nn
import math
from framework.SE_BottleNeck import  V3Bottleneck

class MobileNetV3_OCT2SysD(nn.Module):
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
            [5, 960, 160, True, 'HS', 1],
            [5, 960, 160, True, 'HS', 1],
        ]  # for MobileNet V3 big model
        self.m_bottleneckList = nn.ModuleList()
        for kernel, expandSize, outC, SE, NL, stride in bottleneckConfig:
            self.m_bottleneckList.append(
                V3Bottleneck(inC, outC, kernel=kernel, stride=stride, expandSize=expandSize, SE=SE, NL=NL))
            inC = outC

        self.m_outputConv = nn.Sequential(
            nn.Conv2d(inC, hps.outputChannels, kernel_size=1, stride=1, padding=0, bias=False) #,
            # nn.BatchNorm2d(hps.outputChannels), #*** norm should not be before avgPooling ****
            # nn.Hardswish()
        )

        self._initializeWeights()

    def forward(self,x):
        x = self.m_inputConv(x)
        for bottle in self.m_bottleneckList:
            x = bottle(x)
        x = self.m_outputConv(x)
        return x

    def _initializeWeights(self):
        '''
        refer to https://github.com/xiaomi-automl/MoGA/blob/master/models/MoGA_A.py

        :return:
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()
