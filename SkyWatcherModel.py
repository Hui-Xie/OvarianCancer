from BasicModel import BasicModel
from BuildingBlocks import *
import torch

# SkyWatcher Model, simultaneously train segmentation and treatment response

class SkyWatcherModel(BasicModel):
    def __init__(self, C,  Kr, Kup, inputSize, nDownSamples):
        super().__init__()
        self.m_inputSize = inputSize
        self.m_nDownSamples = nDownSamples
        self.m_bottleNeckSize  = self.getDownSampleSize(self.m_inputSize, self.m_nDownSamples)
        lenBn = self.getProduct(self.m_bottleNeckSize)  # len of BottleNeck

        N = 3  # the number of layer in each building block
        self.m_input = ConvInput(1, C//2, N-1, filterSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # inputSize = output

        self.m_downList = nn.ModuleList()
        for i  in range(self.m_nDownSamples):
            if 0 == i:
                self.m_downList.append(DownBB(C//2, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N))
            else:
                self.m_downList.append(DownBB(C, C, filter1st=(3, 3, 3), stride=(2, 2, 2), nLayers=N))
        # from inputSize 147*281*281, after 6 downsamples, the output size is (1*3*3)

        self.m_fc11   = nn.Sequential(
                       nn.Linear(C*lenBn , C*lenBn//2),
                       nn.InstanceNorm1d(C*lenBn//2),
                       nn.ReLU(inplace=True),
                       nn.Linear(C*lenBn//2, C*lenBn//4),
                       nn.InstanceNorm1d(C*lenBn//4),
                       nn.ReLU(inplace=True),
                       nn.Linear(C*lenBn//4, Kr))

        self.m_upList = nn.ModuleList()
        for i  in range(self.m_nDownSamples):
            if i != self.m_nDownSamples -1:
                self.m_upList.append(UpBB(C, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N))
            else:
                self.m_upList.append(UpBB(C, C//2, filter1st=(3, 3, 3), stride=(2, 2, 2), nLayers=N))
        # from the  bottle neck size (1*3*3), after 6 downsamples, deconv get output size: (127*255*255)

        self.m_upOutput = nn.Conv3d(C//2, Kup, (1,1,1), stride=(1,1,1))

    def encoderForward(self, inputx):
        x = self.m_input(inputx)
        for down in self.m_downList:
            x = down(x)
        # here x is the output at crossing point of sky watcher
        return x

    def responseForward(self, crossingx):
        # xr means x rightside output, or response output
        xr = crossingx
        xr = torch.reshape(xr, (1, xr.numel()))
        xr = self.m_fc11(xr)
        return xr

    def decoderForward(self, crossingx):
        # xup means the output using upList
        xup = crossingx
        for up in self.m_upList:
            xup = up(xup)
        xup = self.m_upOutput(xup)
        return xup


    def forward(self, inputx):
        x = self.encoderForward(inputx)
        xr = self.responseForward(x)
        xup = self.decoderForward(x)
        return xr, xup

