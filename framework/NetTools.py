
import math
import torch.nn as nn
import torch
import numpy as np

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

def construct2DFeatureNet(inputChannels, C, nLayers, inputActivation=True, numConvEachLayer=3, useBatchNorm=True):
    '''
    # downPooling layer is responsible change size of feature map (by MaxPool) and number of filters.
    #  Pooling->ChannelChange->downLayer
    #  input to downPooling0: BxinputChannelsxHxW
    #  output of downLayer0:  BxCxHxW

    :param inputChannels:
    :param C: the channel number of the first layer, or channels list for all layers.
    :param nLayers:
    :param numConvEachLayer: 2 or 3
    :return:
    '''
    if isinstance(C,int):
        CLayers=[]
        for i in range(nLayers):
            CLayers.append(C*pow(2, i))
    elif isinstance(C, list):
        assert len(C) >= nLayers
        CLayers=C
    else:
        print("C type is not correct")
        assert False

    downPoolings = nn.ModuleList()
    downLayers = nn.ModuleList()
    for i in range(nLayers):
        #CPreLayer = C*pow(2, i-1) if i >=1 else C
        #CLayer = C*pow(2, i)  # the channel number in the Downlayer
        CLayer = CLayers[i]

        if 0 == i:
            downPoolings.append(Conv2dBlock(inputChannels, CLayer, activation=inputActivation, useBatchNorm=useBatchNorm))
        else:
            CPreLayer = CLayers[i-1]
            downPoolings.append(nn.Sequential(
                nn.MaxPool2d(2, stride=2, padding=0),
                Conv2dBlock(CPreLayer, CLayer, useBatchNorm=useBatchNorm)
            ))
        if 3 <= numConvEachLayer:
            downLayers.append(nn.Sequential(
                Conv2dBlock(CLayer, CLayer, useBatchNorm=useBatchNorm),
                Conv2dBlock(CLayer, CLayer, useBatchNorm=useBatchNorm),
                Conv2dBlock(CLayer, CLayer, useBatchNorm=useBatchNorm)))
        elif 2 == numConvEachLayer:
            downLayers.append(nn.Sequential(Conv2dBlock(CLayer, CLayer, useBatchNorm=useBatchNorm),
                                            Conv2dBlock(CLayer, CLayer, useBatchNorm=useBatchNorm)))
        else:
            downLayers.append(nn.Sequential(Conv2dBlock(CLayer, CLayer, useBatchNorm=useBatchNorm)))

    return downPoolings, downLayers


def argSoftmax(x, rangeH=None):
    '''

    :param x: in (BatchSize, NumSurface, H, W) dimension, the value is probability (after Softmax) along the height dimension
    :Param range: [H0, H1]
    :return:  mu in size: B,N,1,W
    '''
    device = x.device
    B, N, H, W = x.shape
    if rangeH is None:
        H0 = 0
        H1 = H
    else:
        H0 = rangeH[0]
        H1 = rangeH[1]
        assert (H == int(H1-H0))

    # compute mu
    Y = torch.arange(start=H0, end=H1, step=1.0).view((1, 1, H, 1)).expand(x.size()).to(device=device, dtype=torch.int16)
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

def columnHausdorffDist(data1, data2):
    '''

    :param data1: In BxNxW size in numpy
    :param data2: In BxNxW size in numpy
    :return:  a vector of size N in numpy
    '''
    B,N,W = data1.shape
    return np.amax(np.abs(data1-data2), axis=(0,2)).reshape((N,1))

def computeMuVariance(x, layerMu=None, layerConf=None, rangeH=None): # without square weight
    '''
    Compute the mean and variance along H direction of each surface.

    :param x: in (BatchSize, NumSurface, H, W) dimension, the value is probability (after Softmax) along each Height direction
           LayerMu: the referred surface mu from LayerProb, in size(B,N,W); where N = NumSurface.
           LayerConf: the referred surface confidence from LayerProb, in size(B,N,W)
           rangeH: the maximum mu computed.
    :return: mu:     mean in (BatchSize, NumSurface, W) dimension
             sigma2: variance in (BatchSize, Numsurface, W) dimension
    '''
    A =3.0  # weight factor to balance surfaceMu and LayerMu.

    device = x.device
    B,N,H,W = x.size() # Num is the num of surface for each patient
    if rangeH is None:
        rangeH = H

    # compute mu
    Y = torch.arange(0, rangeH, step=rangeH/H).view((1,1,H,1)).expand(x.size()).to(device=device, dtype=torch.float32)
    # mu = torch.sum(x*Y, dim=-2, keepdim=True)
    # use slice method to compute P*Y
    for b in range(B):
        if 0==b:
            PY = (x[b,]*Y[b,]).unsqueeze(dim=0)
        else:
            PY = torch.cat((PY, (x[b,]*Y[b,]).unsqueeze(dim=0)))
    mu = torch.sum(PY, dim=-2, keepdim=True) # size: B,N,1,W
    del PY  # hope to free memory.

    if (layerMu is not None) and (layerConf is not None):  # consider LayerMu, adjust mu computed by surface only
       assert layerMu.shape == layerConf.shape
       layerMu = layerMu.unsqueeze(dim=-2)
       layerConf = layerConf.unsqueeze(dim=-2)
       mu = (layerMu*layerConf + mu*(A-layerConf))/A

    # compute sigma2 (variance)
    Mu = mu.expand(x.size())

    #sigma2 = torch.sum(x*torch.pow(Y-Mu,2), dim=-2,keepdim=False)
    # this slice method is to avoid using big GPU memory .
    for b in range(B):
        if 0==b:
            sigma2 = torch.sum(x[b,]*torch.pow(Y[b,]-Mu[b,],2), dim=-2,keepdim=False).unsqueeze(dim=0)
        else:
            sigma2 = torch.cat((sigma2, torch.sum(x[b,]*torch.pow(Y[b,]-Mu[b,],2), dim=-2,keepdim=False).unsqueeze(dim=0)))

    # very important, otherwise sigma2 will increase to make the loss small
    # allowing sigma2 back propogation give better test result in the IVUS data.
    # todo: for experiment: /local/vol00/scratch/Users/hxie1/Projects/DeepLearningSeg/OCTMultiSurfaces/testConfig/
    #                      expUnetJHU_Surface_Layer_20200206/expUnetJHU_SurfaceNet_Sigma0_NoBPSigma_20200302_2.yaml
    # for IVUS data, Not backpropagating simga does not give better result;
    # At March 23rd, 2021, sigma2 should not backward in any optmization module.
    # At April 21st, 2021, allow sigma2 backward propagation. Application layers decide how to use sigma2.
    # At optiModel, opt variable is not sigma2, so it is not the case that optiModel will add simga2.
    # sigma2 = sigma2.detach()

    return mu.squeeze(dim=-2),sigma2

