
import math
import torch.nn as nn
import torch

import sys
from framework.ConvBlocks import *

def computeLayerSizeUsingMaxPool2D(H, W, nLayers, kernelSize=2, stride=2, padding=0, dilation=1):
    '''
    use MaxPool2D to change layer size.
    :param W:
    :param H:
    :param nLayers:
    :param kernelSize:
    :param stride:
    :param padding:
    :param dilation:
    :return: a list of layer size of tuples:
             list[i] indicate the outputsize of layer i
    '''
    layerSizeList= []
    layerSizeList.append((H,W))
    for i in range(1, nLayers):
        H =int(math.floor((H+2*padding-dilation*(kernelSize-1)-1)/stride+1))
        W =int(math.floor((W+2*padding-dilation*(kernelSize-1)-1)/stride+1))
        layerSizeList.append((H, W))
    return  layerSizeList

def constructUnet(inputChannels, H, W, C, nLayers):
    layerSizeList = computeLayerSizeUsingMaxPool2D(H, W, nLayers)

    downPoolings = nn.ModuleList()
    downLayers = nn.ModuleList()
    upSamples = nn.ModuleList()
    upLayers = nn.ModuleList()
    # symmetric structure in each layer:
    # downPooling layer is responsible change size of feature map (by MaxPool) and number of filters.
    #  Pooling->ChannelChange->downLayer    ==============     upLayer->UpperSample->ChannelChange
    #  input to downPooling0: BxinputChannelsxHxW
    #  output of upSample0:  BxCxHxW
    for i in range(nLayers):
        CPreLayer = C*pow(2, i-1) if i >=1 else C
        CLayer = C*pow(2, i)  # the channel number in the layer
        if 0 == i:
            downPoolings.append(Conv2dBlock(inputChannels, CLayer))
            upSamples.append(Conv2dBlock(CLayer, CPreLayer))
        else:
            downPoolings.append(nn.Sequential(
                nn.MaxPool2d(2, stride=2, padding=0),
                Conv2dBlock(CPreLayer, CLayer)
            ))
            upSamples.append(nn.Sequential(
                nn.Upsample(size=layerSizeList[i - 1], mode='bilinear'),
                Conv2dBlock(CLayer, CPreLayer)
            ))
        downLayers.append(nn.Sequential(
            Conv2dBlock(CLayer, CLayer), Conv2dBlock(CLayer, CLayer), Conv2dBlock(CLayer, CLayer)))

        upLayers.append(nn.Sequential(
            Conv2dBlock(CLayer, CLayer), Conv2dBlock(CLayer, CLayer), Conv2dBlock(CLayer, CLayer)))

    return downPoolings, downLayers, upSamples, upLayers

def argSoftmax(x):
    '''

    :param x: in (BatchSize, NumSurface, H, W) dimension, the value is probability (after Softmax) along the height dimension
    :param dim:
    :return:  mu in size: B,N,1,W
    '''
    device = x.device
    B, N, H, W = x.shape

    # compute mu
    Y = torch.arange(H).view((1, 1, H, 1)).expand(x.size()).to(device=device, dtype=torch.int16)
    # mu = torch.sum(x*Y, dim=-2, keepdim=True)
    # use slice method to compute P*Y
    for b in range(B):
        if 0 == b:
            PY = (x[b,] * Y[b,]).unsqueeze(dim=0)
        else:
            PY = torch.cat((PY, (x[b,] * Y[b,]).unsqueeze(dim=0)))
    mu = torch.sum(PY, dim=-2, keepdim=False)  # size: B,N,W
    del PY  # hope to free memory.

    return mu
