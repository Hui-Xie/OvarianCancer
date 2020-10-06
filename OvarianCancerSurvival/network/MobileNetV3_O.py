# refer to paper:
# Andrew Howard and Mark Sandler and Grace Chu and Liang-Chieh Chen
# and Bo Chen and Mingxing Tan and Weijun Wang and Yukun Zhu and
# Ruoming Pang and Vijay Vasudevan and Quoc V. Le and Hartwig Adam.
# “Searching for MobileNetV3”. 2019. https://arxiv.org/abs/1905.02244

# this is a version modified for Ovarian cancer project

import torch
import torch.nn as nn
from framework.SE_BottleNeck import V3Bottleneck
import math


class MobileNetV3_O(nn.Module):
    def __init__(self, inputC, outputC, hps=None):
        '''
        This network does not has final FC layer as in original mobileNet v3 paper.
        Applications need to ada specific application heads.

        :param inputC: channel number of input image
               outputC: final number of output tensor
        :output: 1x1280x1x1 tensor.
        '''
        super().__init__()
        self.hps = hps
        inC = 16 # input Channel number for bottleneck
        self.m_inputConv = nn.Sequential(
            # original network stride =2
            nn.Conv2d(inputC, inC, kernel_size=3, stride=2, padding=1, bias=True),
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
        ] # for MobileNet V3 big model
        self.m_bottleneckList = nn.ModuleList()
        for kernel, expandSize, outC, SE, NL, stride in bottleneckConfig:
            self.m_bottleneckList.append(V3Bottleneck(inC, outC, kernel=kernel, stride=stride, expandSize=expandSize, SE=SE, NL=NL))
            inC = outC

        self.m_conv2d_1 = nn.Sequential(
            nn.Conv2d(inC, outputC, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(outputC),
            #nn.Hardswish()   # hardswish may produce value of 1e+21, which lead following std =inf
            nn.ReLU6(inplace=True)
        )

        # after m_conv2d_1, tensor need global mean on H and W dimension.



        # this self.m_conv2d_2 kill all differentiating information.
        '''
        self.m_conv2d_2 = nn.Sequential(
            nn.Conv2d(960, outputC, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU6(inplace=True)
            #nn.Hardswish()

            # Some people says Sigmoid is better for binary 0,1 classification,
            # but sigmoid lead all features to 0 or 1
            # nn.Sigmoid()
            )
        '''

        self.m_dropout = nn.Dropout(p=hps.dropoutRate, inplace=True)


        # application needs to implement classification head outside mobilenet
        self._initializeWeights()



    def forward(self, x):
        B,_,_,_ = x.shape
        x = self.m_inputConv(x)
        for bottle in self.m_bottleneckList:
            x = bottle(x)

        x = self.m_conv2d_1(x)

        if self.hps.useGlobalMean:
            # Traditionally: global average on H,and W dimension, each feature plane
            # global pooling measures the concentration of feature values in each channel
            x = x.mean(dim=(2,3), keepdim=True)  # size: B,960,1,1
        else:
            # Non-traditionally: Use IQR+std on H,and W dimension, each feature plane
            # IQR+std measures the spread of feature values in each channel
            xStd = torch.std(x, dim=(2,3), keepdim=True)  #size: B,hps.outputChannels, 1,1
            xFeatureFlat = x.view(B,self.hps.outputChannels,-1)
            xSorted, _ = torch.sort(xFeatureFlat, dim=-1)
            B,C,N = xSorted.shape
            Q1 = N//4
            Q3 = N*3//4
            # InterQuartile Range
            IQR = (xSorted[:,:,Q3]-xSorted[:,:,Q1]).unsqueeze(dim=-1).unsqueeze(dim=-1)
            x = xStd + IQR

        # x = self.m_conv2d_2(x)  # size: B,outputC,1,1

        x = self.m_dropout(x)

        if torch.isnan(x.sum()) or torch.isinf(x.sum()):  # detect NaN
            print(f"Error: find NaN of x at mobileNet v3 line 177 ")
            assert False

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





