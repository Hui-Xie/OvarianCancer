
import torch.nn as nn
import math
from framework.SE_BottleNeck import *

class MobileNetV3_OCT2SysD(nn.Module):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        inC = 16  # input Channel number for bottleneck

        if hps.inputActivation:
            self.m_inputConv = nn.Sequential(
                nn.Conv2d(hps.inputChannels, inC, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(inC),
                nn.Hardswish()
            )
        else:
            self.m_inputConv = nn.Sequential(
                nn.Conv2d(hps.inputChannels, inC, kernel_size=3, stride=2, padding=1, bias=True),
                nn.BatchNorm2d(inC)
            )

        if hps.mobileNetV3Cfg == "small":
            bottleneckConfig = smallMobileNetV3Config
        else:
            bottleneckConfig = largeMobileNetV3Config

        self.m_bottleneckList = nn.ModuleList()
        for kernel, expandSize, outC, SE, NL, stride in bottleneckConfig:
            self.m_bottleneckList.append(
                V3Bottleneck(inC, outC, kernel=kernel, stride=stride, expandSize=expandSize, SE=SE, NL=NL))
            inC = outC

        self.m_outputConv = nn.Sequential(
            nn.Conv2d(inC, hps.outputChannels, kernel_size=1, stride=1, padding=0, bias=True) #,
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
