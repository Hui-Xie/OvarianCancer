import torch.nn as nn
import torch
import torch.nn.functional as F
import sys

useConvSeq = True

class ConvInput(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        if nLayers < 2:
            print("Error: ConvSeqDecreaseChannels needs at least 2 conve layers.")
            sys.exit(-1)
        self.m_conv1 = nn.Conv2d(inCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1))
        self.m_convBlock = ConvSequential(outCh, outCh, nLayers)

    def forward(self, inputx):
        x = inputx
        x = self.m_conv1(x)
        x = self.m_convBlock(x)
        return x

class ConvOutput(nn.Module):
    def __init__(self, inCh, outCh, nLayers, K):
        """

        :param inCh:
        :param outCh:
        :param K:  final output class before softmax
        """
        super().__init__()
        self.m_convBlock =  ConvSequential(inCh, outCh, nLayers)
        self.m_bn = nn.BatchNorm2d(outCh)
        self.m_conv11= nn.Conv2d(outCh, K, (1, 1), stride=(1, 1))

    def forward(self, inputx, skipInput=None):
        x = inputx if skipInput is None else torch.cat((inputx, skipInput), 1)         # batchsize is in dim 0, so concatenate at dim 1.
        x = self.m_convBlock(x)
        x = self.m_bn(x)
        x = self.m_conv11(x)
        return x

class ConvSeqDecreaseChannels(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        if nLayers < 2:
            print("Error: ConvSeqDecreaseChannels needs at least 2 conve layers.")
            sys.exit(-1)

        self.m_nLayers = nLayers   # the number of conv
        step = (inCh- outCh)//self.m_nLayers
        self.m_convSeq = nn.ModuleList()
        for i in range(self.m_nLayers):
            inChL = inCh - i*step
            outChL = inChL-step if i !=self.m_nLayers-1 else outCh
            self.m_convSeq.append(nn.BatchNorm2d(inChL))
            self.m_convSeq.append(nn.ReLU(inplace=True))
            self.m_convSeq.append(nn.Conv2d(inChL, outChL, (3, 3), stride=(1, 1), padding=(1, 1)))


    def forward(self, inputx, skipInput=None):
        x = inputx if skipInput is None else torch.cat((inputx, skipInput), 1)
        for layer in self.m_convSeq:
            x = layer(x)
        return x


class ConvSequential(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        if nLayers < 2:
            print("Error: ConvSequential needs at least 2 conve layers.")
            sys.exit(-1)

        self.m_nLayers = nLayers   # the number of conv
        self.m_skipStartIndex = 0 if inCh == outCh else 1



        # self.m_conv1 = nn.Conv2d(inCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1))
        # self.m_bn1 =  nn.BatchNorm2d(outCh)
        # if useConvSeq:
        #     self.m_convSeq = nn.Sequential(
        #          nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)),
        #          nn.BatchNorm2d(outCh),
        #          nn.ReLU(inplace=True),
        #          nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)),
        #          nn.BatchNorm2d(outCh),
        #          nn.ReLU(inplace=True)
        #      )
        # else:
        #     self.m_convSeq = nn.ModuleList()
        #     for _ in range(1, self.m_nLayers):
        #         self.m_convSeq.append(nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)))
        #         self.m_convSeq.append(nn.BatchNorm2d(outCh))
        #         self.m_convSeq.append(nn.ReLU(inplace=True))
        self.m_convSeq = nn.ModuleList()
        for i in range(self.m_nLayers):
            if 0 == i:
                self.m_convSeq.append(nn.BatchNorm2d(inCh))
            else:
                self.m_convSeq.append(nn.BatchNorm2d(outCh))

            self.m_convSeq.append(nn.ReLU(inplace=True))

            if 0 == i:
                self.m_convSeq.append(nn.Conv2d(inCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)))
            else:
                self.m_convSeq.append(nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)))


    def forward(self, inputx):
        # x1 = F.relu(self.m_bn1(self.m_conv1(input)), inplace=True)
        # x = x1
        # if useConvSeq:
        #     x = self.m_convSeq(x)
        # else:
        #     for layer in self.m_convSeq:
        #         x = layer(x)
        #
        #
        # # for residual edge
        # if input.shape == x.shape:
        #     return input + x
        # else:
        #     return x1+x

        x = inputx
        if self.m_skipStartIndex == 0:
            x0  = x
        for i, layer in enumerate(self.m_convSeq):
            x = layer(x)
            if 2 == i and self.m_skipStartIndex ==1:
                x0 = x
            if (i+1- self.m_skipStartIndex*3)%6 == 0 and x.shape == x0.shape and i != self.m_nLayers*3 -4:  # a skip connection skip at least 2 layers
                x  = x+x0
                x0 = x
        if (self.m_nLayers - self.m_skipStartIndex) %2  != 0:
            x = x+ x0
        return x

class ConvDense(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        """
        :param inCh: input channels
        :param outCh: output channels
        :param nLayers: total convolutional 3*3 layers,excluding the 1*1 convolutional layer at final
        """
        super().__init__()
        self.m_bnList = nn.ModuleList()
        self.m_reluList = nn.ModuleList()
        self.m_convList = nn.ModuleList()
        self.m_nLayers = nLayers
        k = outCh // nLayers   # growth rate
        midChL = outCh          # the middle channels number after the 1*1 conv inside a conv layer

        for i in range(nLayers):
            inChL  = inCh+ i*k  # inChannels in Layer
            outChL = k if i != nLayers-1 else outCh-k*(nLayers-1)
            self.m_bnList.append(nn.BatchNorm2d(inChL))
            self.m_reluList.append(nn.ReLU(inplace=True))
            self.m_convList.append(nn.Conv2d(inChL, midChL, (1, 1), stride=(1, 1)))
            self.m_bnList.append(nn.BatchNorm2d(midChL))
            self.m_reluList.append(nn.ReLU(inplace=True))
            self.m_convList.append(nn.Conv2d(midChL, outChL, (3, 3), stride=(1, 1), padding=(1, 1)))

        # add 1*1 convoluitonal layer to adjust output channels
        self.m_bnList.append(nn.BatchNorm2d(inCh+outCh))
        self.m_reluList.append(nn.ReLU(inplace=True))
        self.m_convList.append(nn.Conv2d(inCh+outCh, outCh, (1, 1), stride=(1, 1)))

    def forward(self, inputx):
        x = inputx
        for i in range(0, self.m_nLayers*2, 2):
            x0 = x
            x = self.m_convList[i](self.m_reLuList[i](self.m_bnList[i](x)))
            x = self.m_convList[i+1](self.m_reLuList[i+1](self.m_bnList[i+1](x)))
            x = torch.cat((x, x0), 1)
        n = self.m_nLayers*2  # the final element in the ModuleList
        x = self.m_convList[n](self.m_reLuList[n](self.m_bnList[n](x)))
        return x

class Down2dBB(nn.Module): # down sample 2D building block
    def __init__(self, inCh, outCh, filter1st, stride, nLayers=3):
        super().__init__()
        self.m_bn1 = nn.BatchNorm2d(inCh)
        self.m_conv1 = nn.Conv2d(inCh, outCh, filter1st, stride)   # stride to replace maxpooling
        if useConvSeq:
            self.m_convBlock = ConvSequential(outCh, outCh, nLayers)
        else:
            self.m_convBlock = ConvDense(outCh, outCh, nLayers)

    def forward(self, inputx):
        # BN-ReLU- Conv
        x = self.m_conv1(F.relu(self.m_bn1(inputx), inplace=True))
        x = self.m_convBlock(x)
        return x



class Up2dBB(nn.Module): # up sample 2D building block
    def __init__(self, inCh, outCh, filter1st, stride, nLayers= 3):
        super().__init__()
        self.m_bn1 = nn.BatchNorm2d(inCh)
        self.m_convT1 = nn.ConvTranspose2d(inCh, outCh, filter1st, stride)   # stride to replace upsample
        if useConvSeq:
            self.m_convBlock = ConvSequential(outCh, outCh, nLayers)
        else:
            self.m_convBlock = ConvDense(outCh, outCh, nLayers)

    def forward(self, downInput, skipInput=None):
        x = downInput if skipInput is None else torch.cat((downInput, skipInput), 1)         # batchsize is in dim 0, so concatenate at dim 1.
        # BN- ReLU- conv
        x = self.m_convT1(F.relu(self.m_bn1(x), inplace=True))
        x = self.m_convBlock(x)
        return x


