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
    input: tensor(Num_surface, W), both sigma and H are a scalar.
    output: tensor(Num_surace, H, W)
    '''
    if 0 == sigma:
        return []
    device = rawLabels.device
    Mu = rawLabels
    Num_Surface, W = Mu.shape
    Mu = Mu.unsqueeze(dim=-2)
    Mu = Mu.expand((Num_Surface, H, W))
    X = torch.arange(start=0.0, end=H, step=1.0).view((1, H, 1)).to(device, dtype=torch.float32)
    X = X.expand((Num_Surface, H, W))
    pi = math.pi
    G = torch.exp(-(X - Mu) ** 2 / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * pi))

    # as G is discrete, it needs normalization to assure its sum =1.
    # otherwise, it may lead occasionaly negative KLDiv Loss.
    Sum = torch.sum(G, dim=-2, keepdim=True)
    Sum = Sum.expand_as(G)
    G = G/Sum

    return G


def batchGaussianizeLabels(rawLabels, Sigma2, H):
    '''

    :param rawLables: in (B,N,W)
    :param Simga2:  in(B,N,W)
    :param H: a scalar
    :return:
           a tensor: (B,N,H,W)
    '''
    device = rawLabels.device
    Mu = rawLabels
    B, N, W = Mu.shape
    Mu = Mu.unsqueeze(dim=-2)
    Mu = Mu.expand((B, N, H, W)) # size: B,N, H,W
    Sigma2 = Sigma2.unsqueeze(dim=-2)
    Sigma2 = Sigma2.expand_as(Mu)

    X = torch.arange(start=0.0, end=H, step=1.0).view((1, 1, H, 1)).to(device, dtype=torch.float32)
    X = X.expand_as(Mu)
    pi = math.pi
    G = torch.exp(-(X - Mu) ** 2 / (2 * Sigma2)) / torch.sqrt(2 * pi*Sigma2)

    # as G is discrete, it needs normalization to assure its sum =1.
    # otherwise, it may lead occasionaly negative KLDiv Loss.
    Sum = torch.sum(G, dim=-2, keepdim=True)
    Sum = Sum.expand_as(G)
    G = G / Sum

    return G


def updateGaussianWithImageGradient(gaussDistr, imageGradMagnitude, weight=100):
    '''
    newDistr = (1 + weight*imageGradMagnitude)*gaussDistr
    and then normalize along H dimension

    :param gaussDistr: size of (B,N,H,W)
    :param imageGradMagnitude:  size of (B,H,W)
    :param weight:  a float scalar, e.g. 100
    :return: updated gaussDir: size of (B,N,H,W)
    '''
    B,N,H,W = gaussDistr.shape
    grad = imageGradMagnitude
    grad = grad.unsqueeze(dim=1)
    grad = grad.expand((B, N, H, W))  # size: B,N,H,W

    G = (1.0+ weight*grad)*gaussDistr  # size: B,N,H,W

    # as G is discrete, it needs normalization to assure its sum =1.
    # otherwise, it may lead occasionaly negative KLDiv Loss.
    Sum = torch.sum(G, dim=-2, keepdim=True)
    Sum = Sum.expand_as(G)
    G = G / Sum

    return G


def getLayerLabels(surfaceLabels, height):
    '''
    for N surfaces, surface 0 in its exact location marks as 1, surface N-1 in its exact location marks as N.
    region pixel between surface i and surface i+1 marks as i.

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
    layerLabels.scatter_(0, surfaceLabels, surfaceCodes)  #surface 0 in its exact location marks as 1, surface N-1 mark as N.

    for i in range(1,H):
        layerLabels[i,:] = torch.where(0 == layerLabels[i,:], layerLabels[i-1,:], layerLabels[i,:])

    return layerLabels


def layerProb2SurfaceMu(layerProb):
    '''
    From layer prob map to get surface location. it is like a reverse process of getLayerLabels.
    But because layerProb is not strict order, and maybe some layers are lacking, it is more complicated than
    a reverse process of getLayerLabels.


    :param layerProb:  softmax probability of size (B,N+1,H,W), where N is the number of surfaces
    :return: surfaceMu: in size (B,N,W) with float
             surfaceConf: in size(B,N,W) indicate confidence of of each Mu.
    '''
    device = layerProb.device
    B, Nplus1, H, W = layerProb.shape
    N = Nplus1 - 1

    layerMap = torch.argmax(layerProb, dim=1)  # size: (B,H,W) with longTensor of element[0,N]

    # guarantee order continuous and ordered
    oldLayerMap = layerMap.clone() # size: (B,H,W)
    layerMap[:,0,:], _ = torch.min(layerMap, dim=1)
    for i in range(1,H):
        layerMap[:,i,:] = torch.where(layerMap[:,i,:] < layerMap[:,i-1,:],
                                      torch.where(oldLayerMap[:,i,:] != oldLayerMap[:,i-1,:], oldLayerMap[:,i-1,:]+1, layerMap[:,i-1,:]),
                                      layerMap[:,i,:])
    layerMap = torch.where(layerMap >N, N*torch.ones_like(layerMap), layerMap)

    # compute mu's confidence
    surfaceConf = (layerMap == oldLayerMap).sum(dim=1, keepdim=True)/H  # size: B,1, W
    surfaceConf = surfaceConf.expand((B,N,W))

    layerMap0 = layerMap[:,0:-1,:]
    layerMap1 = layerMap[:,1:  ,:]  # size: B,H-1,W
    diff = (layerMap1 -layerMap0)  # size: B,H-1,W; diff>0, indicate its next row is surface.

    diff = torch.cat((torch.zeros((B,1,W),device=device, dtype=torch.long), diff), dim=1)
    # size: B,H,W; if diff>0 indicate current pos is a possible surface

    zeroBHW = torch.zeros_like(diff)
    surfaceMap = torch.where(diff>0, layerMap, zeroBHW)   # ignore negative diff
    # size: B,H,W; where [1,N] indicate possible surface, 0 indicates non-surface location;

    # compress same index
    surfaceMu = torch.zeros((B, N, W), dtype=torch.float, device=device)
    pos = torch.tensor(range(0, H), device=device).view(1, H, 1).expand_as(layerMap)  # size: B,H,W
    for i in range(1, N+1): # N surfaces
        iSurface = torch.where(surfaceMap == i, pos, zeroBHW) # size: B,H,W
        #maybe use average of nonzeros element along dimenions is a better soltuion.
        iSurfaceSum = torch.sum(iSurface, dim=1)
        iSurfaceCount = (iSurface !=0).sum(dim=1) +1e-8
        iSurface = iSurfaceSum/iSurfaceCount
        #iSurface, _ = torch.max(iSurface,dim=1)   # size: B,W; compress same index by choosing max position
        surfaceMu[:,i-1,:] = iSurface

    # fill lack surface
    if (surfaceMu ==0).any():
        surfaceMu[:,N-1,:], _ = torch.max(surfaceMu, dim=1)  # size: B,W
        for i in range(N-2,-1,-1): # surface(N-2) to surface0
            surfaceMu[:,i,:] = torch.where(0 == surfaceMu[:,i,:], surfaceMu[0,i+1,:], surfaceMu[:,i,:] )

    return surfaceMu, surfaceConf

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
        newPolarImage = np.zeros((B,newH,W),dtype=np.float32)
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



