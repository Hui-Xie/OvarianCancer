import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvSequential(nn.Module):
    def __init__(self, inCh, outCh):
        super().__init__()
        self.m_conv1 = nn.Conv2d(inCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1))
        self.m_bn1 =  nn.BatchNorm2d(outCh)
        self.m_convSeq = nn.Sequential(
            nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(outCh),
            nn.ReLU(inplace=True),
            nn.Conv2d(outCh, outCh, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(outCh),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        x1 = F.relu(self.m_bn1(self.m_conv1(input)), inplace=True)
        x2 = self.m_convSeq(x1)
        # for residual edge
        if input.shape == x2.shape:
            return input + x2
        else:
            return x1+x2


class Down2dBB(nn.Module): # down sample 2D building block
    def __init__(self, inCh, outCh, filter1st, stride):
        super().__init__()
        self.m_conv1 = nn.Conv2d(inCh, outCh, filter1st, stride)   # stride to replace maxpooling
        self.m_bn1   = nn.BatchNorm2d(outCh)
        self.m_convSeq = ConvSequential(outCh, outCh)


    def forward(self, input):
        x = F.relu(self.m_bn1(self.m_conv1(input)), inplace=True)
        x = self.m_convSeq(x)
        return x



class Up2dBB(nn.Module): # up sample 2D building block
    def __init__(self, inCh, outCh, filter1st, stride):
        super().__init__()
        self.m_convT1 = nn.ConvTranspose2d(inCh, outCh, filter1st, stride)   # stride to replace upsample
        self.m_bn1   = nn.BatchNorm2d(outCh)
        self.m_convSeq = ConvSequential(outCh, outCh)

    def forward(self, downInput, skipInput=None):
        x = downInput if skipInput is None else torch.cat((downInput, skipInput), 1)         # batchsize is in dim 0, so concatenate at dim 1.
        x = F.relu(self.m_bn1(self.m_convT1(x)),  inplace=True)
        x = self.m_convSeq(x)
        return x


