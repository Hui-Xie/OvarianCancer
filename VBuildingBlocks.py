import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvSequential(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        super().__init__()
        self.m_nLayers = nLayers
        self.m_conv1 = nn.Conv2d(inCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1))
        self.m_bn1 =  nn.BatchNorm2d(outCh)
        # self.m_convSeq = nn.Sequential(
        #     nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(outCh),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)),
        #     nn.BatchNorm2d(outCh),
        #     nn.ReLU(inplace=True)
        # )
        self.m_convSeq = nn.ModuleList()
        for _ in range(1, self.m_nLayers):
            self.m_convSeq.append(nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)))
            self.m_convSeq.append(nn.BatchNorm2d(outCh))
            self.m_convSeq.append(nn.ReLU(inplace=True))

    def forward(self, input):
        x1 = F.relu(self.m_bn1(self.m_conv1(input)), inplace=True)
        x = x1
        for layer in self.m_convSeq:
            x = layer(x)
        # x = self.m_convSeq(x)

        # for residual edge
        if input.shape == x.shape:
            return input + x
        else:
            return x1+x

class ConvDense(nn.Module):
    def __init__(self, inCh, outCh, nLayers):
        """
        the total convolutional layers are nLayers plus one 1*1 convolutional layer to fit final outChannel.
        :param inCh: input channels
        :param outCh: output channels
        :param nLayers: total convolutional layers,except the 1*1 convolutional layer
        """
        super().__init__()
        self.m_convList = nn.ModuleList()
        self.m_bnList = nn.ModuleList()
        self.m_reluList = nn.ModuleList()
        self.m_nLayers = nLayers
        k = outCh // nLayers

        for i in range(nLayers):
            inChL  = inCh+ i*k  # inChannels in Layer
            outChL = k if i != nLayers-1 else outCh-k*(nLayers-1)
            self.m_convList.append(nn.Conv2d(inChL, outChL, (3, 3), stride=(1, 1), padding=(1, 1)))
            self.m_bnList.append(nn.BatchNorm2d(outChL))
            self.m_reluList.append(nn.ReLU(inplace=True))

        # add 1*1 convoluitonal layer to adjust output channels
        self.m_convList.append(nn.Conv2d(inCh+outCh, outCh, (1, 1), stride=(1, 1)))
        self.m_bnList.append(nn.BatchNorm2d(outCh))
        self.m_reluList.append(nn.ReLU(inplace=True))
        self.cuda()

    def forward(self, input):
        x = input
        for i in range(self.m_nLayers):
            x = torch.cat((self.m_reluList[i](self.m_bnList[i](self.m_convList[i](x))), x), 1)
        n = self.m_nLayers  # the final element in the ModuleList
        x = self.m_reluList[n](self.m_bnList[n](self.m_convList[n](x)))
        return x

class Down2dBB(nn.Module): # down sample 2D building block
    def __init__(self, inCh, outCh, filter1st, stride, nLayers=3):
        super().__init__()
        self.m_conv1 = nn.Conv2d(inCh, outCh, filter1st, stride)   # stride to replace maxpooling
        self.m_bn1   = nn.BatchNorm2d(outCh)
        # self.m_convBlock = ConvSequential(outCh, outCh, nLayers)
        self.m_convBlock = ConvDense(outCh, outCh, nLayers)

    def forward(self, input):
        x = F.relu(self.m_bn1(self.m_conv1(input)), inplace=True)
        x = self.m_convBlock(x)
        return x



class Up2dBB(nn.Module): # up sample 2D building block
    def __init__(self, inCh, outCh, filter1st, stride, nLayers= 3):
        super().__init__()
        self.m_convT1 = nn.ConvTranspose2d(inCh, outCh, filter1st, stride)   # stride to replace upsample
        self.m_bn1   = nn.BatchNorm2d(outCh)
        # self.m_convBlock = ConvSequential(outCh, outCh, nLayers)
        self.m_convBlock = ConvDense(outCh, outCh, nLayers)

    def forward(self, downInput, skipInput=None):
        x = downInput if skipInput is None else torch.cat((downInput, skipInput), 1)         # batchsize is in dim 0, so concatenate at dim 1.
        x = F.relu(self.m_bn1(self.m_convT1(x)),  inplace=True)
        x = self.m_convBlock(x)
        return x


