
import torch
import torch.nn as nn

largeMobileNetV3Config = [
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
        ]  # for MobileNet V3 big model

smallMobileNetV3Config = [
            # kernel, expandSize, outputChannel,  SE,   NL,  stride,
            [3, 16, 16, True, 'RE', 2],
            [3, 72, 24, False, 'RE', 2],
            [3, 88, 24, False, 'RE', 1],
            [5, 96, 40, True, 'HS', 2],
            [5, 240, 40, True, 'HS', 1],
            [5, 240, 40, True, 'HS', 1],
            [5, 120, 48, True, 'HS', 1],
            [5, 144, 48, True, 'HS', 1],
            [5, 288, 96, True, 'HS', 2],
            [5, 576, 96, True, 'HS', 1],
            [5, 576, 96, True, 'HS', 1],
        ]   # for MobileNet V3 small model

class SqueezeExcite(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc2Layers = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Hardsigmoid()  # this must be sigmoid, instead of RelU
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
            nn.Conv2d(inC, expandSize, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(expandSize),
            nonlinearLayer(),

            nn.Conv2d(expandSize, expandSize, kernel_size=kernel, stride=stride, padding=padding, groups=expandSize, bias=True),
            nn.BatchNorm2d(expandSize),
            nonlinearLayer(),
            SELayer(expandSize),

            nn.Conv2d(expandSize, outC, kernel_size=1, stride=1, padding=0,  bias=True),
            nn.BatchNorm2d(outC)
        )

    def forward(self, x):
        if self.useResidual:
            return x + self.expandDepthwiseProject(x)
        else:
            return self.expandDepthwiseProject(x)