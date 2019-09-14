# Conv Block
import torch.nn as nn
import torch.nn.functional as F

class Conv3dBlock(nn.Module):
    """
    Convolution 3d Block
    """
    def __init__(self, inChannels, outChannels, poolingLayer=None, convStride=1, useSpectralNorm=False, useLeakyReLU=False):
        super().__init__()

        self.m_poolingLayer = poolingLayer
        self.m_useLeakyReLU = useLeakyReLU

        self.m_conv = nn.Conv3d(inChannels, outChannels, kernel_size=3, stride=convStride, padding=1, bias=True)
        if useSpectralNorm:
            self.m_conv = nn.utils.spectral_norm(self.m_conv)
        self.m_bn = nn.BatchNorm3d(outChannels)

    def forward(self, x):
        if self.m_poolingLayer:
            x = self.m_poolingLayer(x)

        y = self.m_conv(x)
        y = F.relu(self.m_bn(y), inplace=True) if not self.m_useLeakyReLU \
            else F.leaky_relu(self.m_bn(y), inplace=True)

        return y


class Conv2dBlock(nn.Module):
    """
    Convolution 2d Block
    """

    def __init__(self, inChannels, outChannels, poolingLayer=None, convStride=1, useSpectralNorm=False,
                 useLeakyReLU=False):
        super().__init__()

        self.m_poolingLayer = poolingLayer
        self.m_useLeakyReLU = useLeakyReLU

        self.m_conv = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=convStride, padding=1, bias=True)
        if useSpectralNorm:
            self.m_conv = nn.utils.spectral_norm(self.m_conv)
        self.m_bn = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        if self.m_poolingLayer:
            x = self.m_poolingLayer(x)

        y = self.m_conv(x)
        y = F.relu(self.m_bn(y), inplace=True) if not self.m_useLeakyReLU \
            else F.leaky_relu(self.m_bn(y), inplace=True)

        return y
