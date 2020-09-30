
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