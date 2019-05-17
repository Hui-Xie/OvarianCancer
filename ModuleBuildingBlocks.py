import torch.nn as nn
import torch
import sys

useSkip2Residual = True       # use residual module in each building block, otherwise use DenseBlock
useBnReConvOrder = False       # use Bn-ReLU-Conv2d order in each layer, otherwise use Conv2d-Bn-ReLU order

class BN_ReLU_Conv2d(nn.Module):
    def __init__(self, inCh, outCh, filterSize=(3,3), stride=(1, 1), padding=(1,1), order= True):
        super().__init__()
        self.m_useBnReConvOrder = order
        if filterSize == (1,1):
            padding = (0,0)
        if useBnReConvOrder:
            self.m_bn1 = nn.BatchNorm2d(inCh)
        else:
            self.m_bn1 = nn.BatchNorm2d(outCh)
        self.m_relu1 = nn.ReLU(inplace=True)
        self.m_conv1 = nn.Conv2d(inCh, outCh, filterSize, stride, padding)

    def forward(self, inputx):
        if self.m_useBnReConvOrder:
            x = self.m_conv1(self.m_relu1(self.m_bn1(inputx)))
        else:
            x = self.m_relu1(self.m_bn1(self.m_conv1(inputx)))
        return x

class BN_ReLU_ConvT2d(nn.Module):  # ConvTransposed
    def __init__(self, inCh, outCh, filterSize=(3,3), stride=(1, 1), order = True):
        super().__init__()
        self.m_useBnReConvOrder = order
        if useBnReConvOrder:
            self.m_bn1 = nn.BatchNorm2d(inCh)
        else:
            self.m_bn1 = nn.BatchNorm2d(outCh)
        self.m_relu1 = nn.ReLU(inplace=True)
        self.m_convT1 = nn.ConvTranspose2d(inCh, outCh, filterSize, stride)

    def forward(self, inputx):
        if self.m_useBnReConvOrder:
            x = self.m_convT1(self.m_relu1(self.m_bn1(inputx)))
        else:
            x = self.m_relu1(self.m_bn1(self.m_convT1(inputx)))
        return x


class ConvDecreaseChannels(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        if nLayers < 2:
            print("Error: ConvDecreaseChannels needs at least 2 conv layers.")
            sys.exit(-1)

        self.m_nLayers = nLayers   # the number of conv
        step = (inCh- outCh)//self.m_nLayers
        self.m_convBlocks = nn.ModuleList()
        for i in range(self.m_nLayers):
            inChL = inCh - i*step
            outChL = inChL-step if i !=self.m_nLayers-1 else outCh
            self.m_convBlocks.append(BN_ReLU_Conv2d(inChL, outChL, filterSize=(3,3), stride=(1, 1), padding=(1, 1), order=useBnReConvOrder))

    def forward(self, inputx, skipInput=None):
        x = inputx if skipInput is None else torch.cat((inputx, skipInput), 1)
        for block in self.m_convBlocks:
            x = block(x)
        return x


class Skip2Convs(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        if nLayers < 2:
            print("Error: ConvResidual needs at least 2 conv layers.")
            sys.exit(-1)

        self.m_nLayers = nLayers   # the number of conv
        self.m_skipStartIndex = 0 if inCh == outCh else 1
        self.m_convBlocks = nn.ModuleList()
        for i in range(self.m_nLayers):
            if 0 == i:
                self.m_convBlocks.append(BN_ReLU_Conv2d(inCh, outCh, order=useBnReConvOrder))
            else:
                self.m_convBlocks.append(BN_ReLU_Conv2d(outCh, outCh, order=useBnReConvOrder))

    def forward(self, inputx):
        x = inputx
        if self.m_skipStartIndex == 0:
            x0  = x
        for i, block in enumerate(self.m_convBlocks):
            x = block(x)
            if 0 == i and self.m_skipStartIndex ==1:
                x0 = x
            if i > 0 and (i+1- self.m_skipStartIndex)%2 == 0 and x.shape == x0.shape and i != self.m_nLayers -2:  # a skip connection skips at least 2 layers
                x  = x+x0
                x0 = x

        if (self.m_nLayers - self.m_skipStartIndex) %2 != 0 and x.shape == x0.shape:
            x = x+ x0
        return x

class Conv33_11Residual(nn.Module):
    def __init__(self, inCh, outCh):
        super().__init__()
        self.m_33conv = BN_ReLU_Conv2d(inCh, outCh, filterSize=(3,3), stride=(1, 1), padding=(1,1), order=useBnReConvOrder)
        self.m_11conv = BN_ReLU_Conv2d(inCh, outCh, filterSize=(1,1), stride=(1, 1), padding=(0,0), order=useBnReConvOrder)

    def forward(self, inputx):
        x = self.m_33conv(inputx) + self.m_11conv(inputx)
        return x

class ResPath(nn.Module):
    r"""
    Please refer paper: MultiResUNet : Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation
    link: https://arxiv.org/abs/1902.04049

    """
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        self.m_convBlocks = nn.ModuleList()
        self.m_nLayers = nLayers
        for i in range(nLayers):
            if i==0:
                self.m_convBlocks.append(Conv33_11Residual(inCh,outCh))
            else:
                self.m_convBlocks.append(Conv33_11Residual(outCh, outCh))

    def forward(self, inputx):
        x = inputx
        for i in range(self.m_nLayers):
            x = self.m_convBlocks[i](x)
        return x


class ConvDense(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        """
        :param inCh: input channels
        :param outCh: output channels
        :param nLayers: total convolutional 3*3 layers,excluding the 1*1 convolutional layer at final
        """
        super().__init__()
        self.m_convBlocks = nn.ModuleList()
        self.m_nLayers = nLayers
        k = outCh // nLayers   # growth rate
        midChL = outCh          # the middle channels number after the 1*1 conv inside a conv layer

        for i in range(nLayers):
            inChL  = inCh+ i*k  # inChannels in Layer
            outChL = k if i != nLayers-1 else outCh-k*(nLayers-1)
            self.m_convBlocks.append(BN_ReLU_Conv2d(inChL, midChL,  filterSize=(1,1), stride=(1,1), padding=(0,0), order=useBnReConvOrder))
            self.m_convBlocks.append(BN_ReLU_Conv2d(midChL, outChL, filterSize=(3, 3), stride=(1, 1), padding=(1, 1), order=useBnReConvOrder))

        # add 1*1 convoluitonal layer to adjust output channels
        self.m_convBlocks.append(BN_ReLU_Conv2d(inCh+outCh, outCh, (1, 1), stride=(1, 1), padding=(0,0), order=useBnReConvOrder))

    def forward(self, inputx):
        x = inputx
        for i in range(0, self.m_nLayers*2, 2):
            x0 = x
            x = self.m_convBlocks[i](x)
            x = self.m_convBlocks[i+1](x)
            x = torch.cat((x, x0), 1)
        n = self.m_nLayers*2  # the final element in the ModuleList
        x =  self.m_convBlocks[n](x)
        return x

class ConvBuildingBlock(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        if useSkip2Residual:   # use residual links
            self.m_convBlock = Skip2Convs(inCh, outCh, nLayers)
        else:             # use Dense Links
            self.m_convBlock = ConvDense(inCh, outCh, nLayers)

    def forward(self, inputx):
        return self.m_convBlock(inputx)


class ConvInput(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        if nLayers < 2:
            print("Error: ConvInput needs at least 2 conv layers.")
            sys.exit(-1)
        self.m_convLayer = BN_ReLU_Conv2d(inCh, outCh, filterSize=(3, 3), stride=(1, 1), padding=(1, 1), order=useBnReConvOrder)
        self.m_convBlocks = ConvBuildingBlock(outCh, outCh, nLayers)

    def forward(self, inputx):
        x = inputx
        x = self.m_convLayer(x)
        x = self.m_convBlocks(x)
        return x

class ConvOutput(nn.Module):
    def __init__(self, inCh, outCh, nLayers, K):
        """

        :param inCh:
        :param outCh:
        :param K:  final output class before softmax
        """
        super().__init__()
        self.m_convBlocks =  ConvBuildingBlock(inCh, outCh, nLayers)
        self.m_bn = nn.BatchNorm2d(outCh)
        self.m_conv11 = nn.Conv2d(outCh, K, (1, 1), stride=(1, 1))

    def forward(self, inputx, skipInput=None):
        x = inputx if skipInput is None else torch.cat((inputx, skipInput), 1)         # batchsize is in dim 0, so concatenate at dim 1.
        x = self.m_convBlocks(x)
        if useBnReConvOrder:
            x = self.m_conv11(self.m_bn(x))
        else:
            x = self.m_conv11(x)   # no need bn and relu
        return x

class Down2dBB(nn.Module): # down sample 2D building block
    def __init__(self, inCh, outCh, filter1st, stride, nLayers):
        super().__init__()
        self.m_downLayer = BN_ReLU_Conv2d(inCh, outCh, filterSize=filter1st, stride=stride, padding=(0, 0), order=useBnReConvOrder)
        self.m_convBlocks = ConvBuildingBlock(outCh, outCh, nLayers)

    def forward(self, inputx):
        x = self.m_downLayer(inputx)
        x = self.m_convBlocks(x)
        return x


class Up2dBB(nn.Module): # up sample 2D building block
    def __init__(self, inCh, outCh, filter1st, stride, nLayers):
        super().__init__()
        self.m_upLayer = BN_ReLU_ConvT2d(inCh, outCh, filterSize=filter1st, stride=stride, order=useBnReConvOrder)
        self.m_convBlocks = ConvBuildingBlock(outCh, outCh, nLayers)


    def forward(self, downInput, skipInput=None):
        x = downInput if skipInput is None else torch.cat((downInput, skipInput), 1)         # batchsize is in dim 0, so concatenate at dim 1.
        x = self.m_upLayer(x)
        x = self.m_convBlocks(x)
        return x


