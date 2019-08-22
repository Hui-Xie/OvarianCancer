# DeformConvBlock
import torch.nn as nn
from DeformConv2d import DeformConv2d
import torch.nn.functional as F

class DeformConvBlock(nn.Module):
    """
    3 deformConv2ds concatenate, plus an identity connection.
    """
    def __init__(self, inChannels, outChannels, poolingLayer=None, convStride=1, useLeakyReLU=False):
        super().__init__()

        self.m_poolingLayer = poolingLayer
        self.m_useLeakyReLU = useLeakyReLU

        self.m_deformConv1 = DeformConv2d(inChannels, inChannels, kernel_size=3, padding=1, stride=1, bias=True, modulation=True)
        self.m_BN1 = nn.BatchNorm2d(inChannels)

        self.m_deformConv2 = DeformConv2d(inChannels, inChannels, kernel_size=3, padding=1, stride=convStride, bias=True, modulation=True)
        self.m_BN2 = nn.BatchNorm2d(inChannels)

        self.m_deformConv3 = DeformConv2d(inChannels, outChannels, kernel_size=3, padding=1, stride=1, bias=True, modulation=True)
        self.m_BN3 = nn.BatchNorm2d(inChannels)

        if inChannels != outChannels or convStride != 1:
            self.m_identityConv = DeformConv2d(inChannels, outChannels, kernel_size=1 if convStride == 1 else 3,
                                                                             padding=0 if convStride == 1 else 1, stride=convStride, bias=True, modulation=True)
            self.m_identityBN = nn.BatchNorm2d(outChannels)
        else:
            self.m_identityConv = None
            self.m_identityBN = None

    def forward(self, x):
        if self.m_poolingLayer:
            x = self.m_poolingLayer(x)

        y = self.m_deformConv1(x)
        y = F.relu(self.m_BN1(y), inplace=True) if not self.m_useLeakyReLU \
            else F.leaky_relu(self.m_BN1(y), inplace=True)

        y = self.m_deformConv2(y)
        y = F.relu(self.m_BN2(y), inplace=True)  if not self.m_useLeakyReLU \
            else F.leaky_relu(self.m_BN2(y),inplace=True)

        y = self.m_deformConv3(y)
        y = self.m_BN3(y)

        if self.m_identityConv and self.m_identityBN:
            x = self.m_identityConv(x)
            x = self.m_identityBN(x)

        return F.relu(x + y, inplace=True)  if not self.m_useLeakyReLU \
               else F.leaky_relu(x + y, inplace=True)
