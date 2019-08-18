# ResNeXt Block
import torch.nn as nn
import torch.nn.functional as F

class ResNeXtBlock(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, inChannels, outChannels, nGroups, poolingLayer=None, convStride=1, useSpectralNorm=False, useLeakyReLU=False):
        super().__init__()

        if inChannels % nGroups !=0:
            print(f"Error: inChannels {inChannels} must be integer times of nGroups{nGroups}.")

        self.m_poolingLayer = poolingLayer
        self.m_useLeakyReLU = useLeakyReLU

        self.m_reduceConv = nn.Conv2d(inChannels, inChannels, kernel_size=1, stride=1, padding=0, bias=True)
        if useSpectralNorm:
            self.m_reduceConv = nn.utils.spectral_norm(self.m_reduceConv)
        self.m_reduceBN = nn.BatchNorm2d(inChannels)

        self.m_groupConv = nn.Conv2d(inChannels, inChannels, kernel_size=3, stride=convStride, padding=1, groups=nGroups, bias=True)
        if useSpectralNorm:
            self.m_groupConv = nn.utils.spectral_norm(self.m_groupConv)
        self.m_groupBN = nn.BatchNorm2d(inChannels)

        self.m_expandConv = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0, bias=True)
        if useSpectralNorm:
            self.m_expandConv = nn.utils.spectral_norm(self.m_expandConv)
        self.m_expandBN = nn.BatchNorm2d(outChannels)

        if inChannels != outChannels or convStride != 1:
            self.m_identityConv = nn.Conv2d(inChannels, outChannels, kernel_size=1 if convStride == 1 else 3, stride=convStride, padding=0 if convStride == 1 else 1, bias=True)
            if useSpectralNorm:
                self.m_identityConv = nn.utils.spectral_norm(self.m_identityConv)
            self.m_identityBN = nn.BatchNorm2d(outChannels)
        else:
            self.m_identityConv = None
            self.m_identityBN = None

    def forward(self, x):
        if self.m_poolingLayer:
            x = self.m_poolingLayer(x)

        y = self.m_reduceConv(x)
        y = F.relu(self.m_reduceBN(y), inplace=True) if not self.m_useLeakyReLU \
            else F.leaky_relu(self.m_reduceBN(y), inplace=True)

        y = self.m_groupConv(y)
        y = F.relu(self.m_groupBN(y), inplace=True)  if not self.m_useLeakyReLU \
            else F.leaky_relu(self.m_groupBN(y), inplace=True)

        y = self.m_expandConv(y)
        y = self.m_expandBN(y)

        if self.m_identityConv and self.m_identityBN:
            x = self.m_identityConv(x)
            x = self.m_identityBN(x)

        return F.relu(x + y, inplace=True)  if not self.m_useLeakyReLU \
               else F.leaky_relu(x + y, inplace=True)
