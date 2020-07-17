# Conv Block

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Conv3dBlock(nn.Module):
    """
    Convolution 3d Block
    """
    def __init__(self, inChannels, outChannels, convStride=1, useSpectralNorm=False, useLeakyReLU=False, kernelSize=3, padding=1):
        super().__init__()

        self.m_useLeakyReLU = useLeakyReLU

        self.m_conv = nn.Conv3d(inChannels, outChannels, kernel_size=kernelSize, stride=convStride, padding=padding, bias=True)
        if useSpectralNorm:
            self.m_conv = nn.utils.spectral_norm(self.m_conv)
        # self.m_norm = nn.BatchNorm3d(outChannels)
        self.m_norm = nn.InstanceNorm3d(outChannels)

    def forward(self, x):
        y = self.m_conv(x)

        featureMapSize = np.prod(y.shape[-3:])
        if featureMapSize > 8:
            # with Normalization
            y = F.relu(self.m_norm(y), inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(self.m_norm(y), inplace=True)
        else:
            # without Normalization
            y = F.relu(y, inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(y, inplace=True)
        return y

class Deconv3dBlock(nn.Module):
    """
    DeConvolution 3d Block
    """
    def __init__(self, inChannels, outChannels, convStride=1, useSpectralNorm=False, useLeakyReLU=False, kernelSize=3, padding=1):
        super().__init__()

        self.m_useLeakyReLU = useLeakyReLU

        self.m_conv = nn.ConvTranspose3d(inChannels, outChannels, kernel_size=kernelSize, stride=convStride, padding=padding, bias=True)
        if useSpectralNorm:
            self.m_conv = nn.utils.spectral_norm(self.m_conv)
        # self.m_norm = nn.BatchNorm3d(outChannels)
        self.m_norm = nn.InstanceNorm3d(outChannels)

    def forward(self, x):
        y = self.m_conv(x)

        featureMapSize = np.prod(y.shape[-3:])
        if featureMapSize > 8:
            # with Normalization
            y = F.relu(self.m_norm(y), inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(self.m_norm(y), inplace=True)
        else:
            # without Normalization
            y = F.relu(y, inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(y, inplace=True)
        return y

class Conv2dBlock(nn.Module):
    """
    Convolution 2d Block
    """

    def __init__(self, inChannels, outChannels, convStride=1, useSpectralNorm=False,
                 useLeakyReLU=False, kernelSize=3, padding=1, dilation=1):
        super().__init__()

        self.m_useLeakyReLU = useLeakyReLU

        self.m_conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=convStride, padding=padding, dilation=dilation, bias=True)
        if useSpectralNorm:
            self.m_conv = nn.utils.spectral_norm(self.m_conv)
        # self.m_norm = nn.BatchNorm2d(outChannels)
        self.m_norm = nn.InstanceNorm2d(outChannels)  # Instance Norm applies on per channel.

    def forward(self, x):
        y = self.m_conv(x)

        featureMapSize = np.prod(y.shape[-2:])
        if featureMapSize > 4:
            # with Normalization
            y = F.relu(self.m_norm(y), inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(self.m_norm(y), inplace=True)
        else:
            # without Normalization
            y = F.relu(y, inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(y, inplace=True)

        return y


class Deconv2dBlock(nn.Module):
    """
    DeConvolution 2d Block
    """

    def __init__(self, inChannels, outChannels, convStride=1, useSpectralNorm=False,
                 useLeakyReLU=False, kernelSize=3, padding=0):
        super().__init__()

        self.m_useLeakyReLU = useLeakyReLU

        self.m_conv = nn.ConvTranspose2d(inChannels, outChannels, kernel_size=kernelSize, stride=convStride, padding=padding, bias=True)
        if useSpectralNorm:
            self.m_conv = nn.utils.spectral_norm(self.m_conv)
        # self.m_norm = nn.BatchNorm2d(outChannels)
        self.m_norm = nn.InstanceNorm2d(outChannels)  # Instance Norm applies on per channel.

    def forward(self, x):
        y = self.m_conv(x)

        featureMapSize = np.prod(y.shape[-2:])
        if featureMapSize > 4:
            # with Normalization
            y = F.relu(self.m_norm(y), inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(self.m_norm(y), inplace=True)
        else:
            # without Normalization
            y = F.relu(y, inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(y, inplace=True)

        return y


class LinearBlock(nn.Module):
    """
    Linear block
    """

    def __init__(self, inFeatures, outFeatures, useLeakyReLU=False, bias=True, useNonLinearActivation=True, normModule=None):
        super().__init__()

        self.m_useLeakyReLU = useLeakyReLU
        self.m_useNonLinearActivation = useNonLinearActivation

        self.m_linear = nn.Linear(inFeatures, outFeatures, bias=bias)
        self.m_norm = normModule

    def forward(self, x):
        y = self.m_linear(x)

        featureMapSize = y.shape[-1]
        if self.m_norm is not None:
            y = self.m_norm(y)
        if self.m_useNonLinearActivation:
            y = F.relu(y, inplace=True) if not self.m_useLeakyReLU \
                else F.leaky_relu(y, inplace=True)

        return y

