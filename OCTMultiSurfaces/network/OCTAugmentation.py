# define OCT Augmentation functions

import torch
import math
import cv2
import numpy as np

def polarImageLabelRotate_Tensor(polarImage, polarLabel, rotation=0):
    '''

    :param polarImage: size of (H,W) or (B,H,W)
    :param polarLabel: in size of (C,W) or (B,C,W)
    :param rotation: in integer degree of [0,360], negative indicates reverse direction
    :return: (rotated polarImage,rotated polarLabel) same size with input
    '''
    assert polarLabel.dim() == polarLabel.dim()
    assert polarImage.shape[-1] == polarLabel.shape[-1]
    rotation = rotation % 360
    if 0 != rotation:
        polarImage = torch.roll(polarImage, rotation, dims=-1)
        polarLabel = torch.roll(polarLabel, rotation, dims=-1)
    return (polarImage, polarLabel)

def polarLabelRotate_Tensor(polarLabel, rotation=0):
    '''

    :param polarLabel: in size of (C,w) or (B,C,W)
    :param rotation: in integer degree of [0,360], negative indicates reverse direction
    :return: (rotated polarImage,rotated polarLabel) same size with input
    '''
    rotation = rotation % 360
    if 0 != rotation:
        polarLabel = torch.roll(polarLabel, rotation, dims=-1)
    return polarLabel


def gaussianizeLabels(rawLabels, sigma, H):
    '''
    input: tensor(Num_surface, W)
    output: tensor(Num_surace, H, W)
    '''
    device = rawLabels.device
    Mu = rawLabels
    Num_Surface, W = Mu.shape
    Mu = Mu.unsqueeze(dim=-2)
    Mu = Mu.expand((Num_Surface, H, W))
    X = torch.arange(start=0.0, end=H, step=1.0).view((H, 1)).to(device, dtype=torch.float32)
    X = X.expand((Num_Surface, H, W))
    pi = math.pi
    G = torch.exp(-(X - Mu) ** 2 / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * pi))
    return G


def getLayerLabels(surfaceLabels, height):
    '''

    :param surfaceLabels: float tensor in size of (N,W) where N is the number of surfaces.
            height: original image height
    :return: layerLabels: long tensor in size of (H, W) in which each element is long integer  of [0,N] indicating belonging layer

    '''
    H = height
    device = surfaceLabels.device
    N, W = surfaceLabels.shape  # N is the number of surface
    layerLabels = torch.zeros((H, W), dtype=torch.long, device=device)
    surfaceLabels = (surfaceLabels + 0.5).long()  # let surface height match grid
    surfaceCodes = torch.tensor(range(1, N + 1), device=device).unsqueeze(dim=-1).expand_as(surfaceLabels)
    layerLabels.scatter_(0, surfaceLabels, surfaceCodes)

    for i in range(1,H):
        layerLabels[i,:] = torch.where(0 == layerLabels[i,:], layerLabels[i-1,:], layerLabels[i,:])

    return layerLabels

def lacePolarImageLabel(data,label,lacingWidth):
    '''
    for single polar image and label, lace both ends with rotation information.
    in order to avoid inaccurate segmentation at image boundary

    :param data: H*W in Tensor format
    :param label: C*W, where C is the number of contour
    :param lacingWidth: integer
    :return:
    '''
    H,W = data.shape
    assert lacingWidth<W
    data =torch.cat((data[:,-lacingWidth:],data, data[:,:lacingWidth]),dim=1)
    assert data.shape[1] == W+2*lacingWidth
    label = torch.cat((label[:,-lacingWidth:],label, label[:,:lacingWidth]),dim=1)
    return data, label

def delacePolarImageLabel(data,label,lacingWidth):
    '''
    suport batch and single polar image and label

    :param data: data: H*W in Tensor format
    :param label: C*W, where C is the number of contour
    :param lacingWidth: integer
    :return:
    '''
    if 2 == data.dim():
        H,W = data.shape
        newW = W -2*lacingWidth
        assert newW > 0
        data  = data[:, lacingWidth:newW+lacingWidth]
        label = label[:, lacingWidth:newW+lacingWidth]
    elif 3 == data.dim():
        B,H,W = data.shape
        newW = W - 2 * lacingWidth
        assert newW > 0
        data = data[:, :, lacingWidth:newW + lacingWidth]
        label = label[:, :, lacingWidth:newW + lacingWidth]
    else:
        print("delacePolarImageLabel currently does not support >=4 dim tensor")
        assert False

    return data, label

def delacePolarLabel(label, lacingWidth):
    if 2 == label.dim():
        H,W = label.shape
        newW = W -2*lacingWidth
        assert newW > 0
        label = label[:, lacingWidth:newW+lacingWidth]
    elif 3 == label.dim():
        B,H,W = label.shape
        newW = W - 2 * lacingWidth
        assert newW > 0
        label = label[:, :, lacingWidth:newW + lacingWidth]
    else:
        print("delacePolarLabel currently does not support >=4 dim tensor")
        assert False

    return label

def scalePolarImage(polarImage, scaleNumerator, scaleDenominator):
    '''
    scaling the radial of polarImage equals sacling X and Y or cartesian image.
    result = R * scaleNumerator//scaleDenominator, where scale factor is better for integer.

    :param polarImage: in (H,W) or (B,H,W)size in Tensor
    :param scaleNumerator: a integer
    :param scaleDenominator: a integer
           both scaleFactor are better integer dividable by orginal value
    :return:
    '''
    device = polarImage.device
    dim = polarImage.dim()
    if 2 == dim:
        H,W = polarImage.shape  # H is radial, and W is angular
        newH = int(H*scaleNumerator/scaleDenominator +0.5)
        polarImage = polarImage.cpu().numpy()
        newPolarImage = cv2.resize(polarImage, (W,newH),interpolation=cv2.INTER_CUBIC)  #cv2 dim is first W,and then Height
        newPolarImage = torch.from_numpy(newPolarImage).to(device)
    elif 3 == dim:
        B,H,W = polarImage.shape
        newH = int(H*scaleNumerator/scaleDenominator +0.5)
        newPolarImage = np.zeros((B,H,W),dtype=torch.float32)
        for b in range(B):
            temp = polarImage[b].cpu().numpy()
            newPolarImage[b] = cv2.resize(temp, (W,newH), interpolation=cv2.INTER_CUBIC)
        newPolarImage = torch.from_numpy(newPolarImage).to(device)
    else:
        print("scalePolarImageRadial does not support dimension >= 4")
        assert False

    return newPolarImage

def scalePolarLabel(polarLabel, scaleNumerator, scaleDenominator):
    '''
    result = R * scaleNumerator/scaleDenominator, where scale factor is better for integer.

    :param polarLabel: in C*W or B*C*W size
    :param scaleNumerator: a integer
    :param scaleDenominator: a integer
    :return:
    '''
    return polarLabel*scaleNumerator/scaleDenominator



