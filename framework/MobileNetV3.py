# refer to paper:
# Andrew Howard and Mark Sandler and Grace Chu and Liang-Chieh Chen
# and Bo Chen and Mingxing Tan and Weijun Wang and Yukun Zhu and
# Ruoming Pang and Vijay Vasudevan and Quoc V. Le and Hartwig Adam.
# “Searching for MobileNetV3”. 2019. https://arxiv.org/abs/1905.02244

# global mean at B,H, and W dimension.

import torch
import torch.nn as nn

class SqueezeExcite(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc2Layers = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        y = self.globalAvgPool(x)
        y = self.fc2Layers(y)
        return torch.mul(x, y) # broadcasted multiplication


class V3Bottleneck(nn.Module):
    def __init__(self, inC, outC, kernel=3, stride=1, expandSize=16, SE=False, NL='RE'):
        super().__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.useResidual = (stride == 1 and inC == outC)

        if NL == 'RE':
            nonlinearLayer  = nn.ReLU6
        elif NL == 'HS':
            nonlinearLayer = nn.Hardswish
        else:
            print(f"Incorrect NonLinear parameter in V3Bottleneck")
            assert False

        if SE:
            SELayer = SqueezeExcite
        else:
            SELayer = nn.Identity

        self.expandDepthwiseProject = nn.Sequential(
            nn.Conv2d(inC, expandSize, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expandSize),
            nonlinearLayer(),

            nn.Conv2d(expandSize, expandSize, kernel_size=kernel, stride=stride, padding=padding, groups=expandSize, bias=False),
            nn.BatchNorm2d(expandSize),
            nonlinearLayer(),
            SELayer(expandSize),

            nn.Conv2d(expandSize, outC, kernel_size=1, stride=1, padding=0,  bias=False),
            nn.BatchNorm2d(outC)
        )

    def forward(self, x):
        if self.useResidual:
            return x + self.expandDepthwiseProject(x)
        else:
            return self.expandDepthwiseProject(x)


class MobileNetV3(nn.Module):
    def __init__(self, inputC):
        '''
        This network does not has final FC layer as in original mobileNet v3 paper.
        Applications need to ada specific application heads.

        :param inputC: channel number of input image
        :output: 1x1280x1x1 tensor.
        '''
        super().__init__()
        inC = 16 # input Channel number for bottleneck
        self.m_inputConv = nn.Sequential(
            # original network stride =2
            # nn.Conv2d(inputC, inC, kernel_size=3, stride=2, padding=1, bias=False),

            # in order to reduce memory
            nn.Conv2d(inputC, inC, kernel_size=3, stride=3, padding=1, bias=False),

            nn.BatchNorm2d(inC),
            nn.Hardswish()
        )

        bottleneckConfig = [
            # kernel, expandSize, outputChannel,  SE,   NL,  stride,
            [3, 16, 16, False, 'RE', 1],
            [3, 64, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1],
            [5, 72, 40, True, 'RE', 2],
            [5, 120, 40, True, 'RE', 1],
            [5, 120, 40, True, 'RE', 1],
            [3, 240, 80, False, 'HS', 2],
            [3, 200, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 184, 80, False, 'HS', 1],
            [3, 480, 112, True, 'HS', 1],
            [3, 672, 112, True, 'HS', 1],
            [5, 672, 160, True, 'HS', 2],
            [5, 960, 160, True, 'HS', 1],
            [5, 960, 160, True, 'HS', 1],
        ] # for MobileNet V3 big model
        self.m_bottleneckList = nn.ModuleList()
        for kernel, expandSize, outC, SE, NL, stride in bottleneckConfig:
            self.m_bottleneckList.append(V3Bottleneck(inC, outC, kernel=kernel, stride=stride, expandSize=expandSize, SE=SE, NL=NL))
            inC = outC

        self.m_conv2d_1 = nn.Sequential(
            nn.Conv2d(inC, 960, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish()
        )

        # after m_conv2d_1, tensor need global mean on B,H and W dimension.

        #self.m_conv2d_2 = nn.Sequential(
        #    nn.Conv2d(960, 1280, kernel_size=1, stride=1, padding=0, bias=False),
        #    nn.Hardswish()
        #)

        #self.m_conv2d_3 = nn.Sequential(
        #    nn.Conv2d(1280, outputK, kernel_size=1, stride=1, padding=0, bias=False)
        #)


    def forward(self, x):
        x = self.m_inputConv(x)
        for bottle in self.m_bottleneckList:
            x = bottle(x)

        x = self.m_conv2d_1(x)

        # global average on B,H,and W.
        # x = x.mean(dim=(0,2,3), keepdim=True)
        # return x


        # introduce mu*sigma to enlarge feature
        globalAvg  = x.mean(dim=(2,3), keepdim=True)  # size: B,C,1,1
        batchStd = globalAvg.std(dim=(0,),keepdim=True)  # size: 1,C,1,1

        # global average on B,H,and W.
        x = x.mean(dim=(0, 2, 3), keepdim=True)

        return x*batchStd  #size: 1,C,1,1





