from BasicModel import BasicModel
from BuildingBlocks import *
import torch

# SkyWatcher Model, simultaneously train segmentation and treatment response

class SkyWatcherModel(BasicModel):
    def __init__(self, C,  Kr, Kup, inputSize, nDownSamples):
        super().__init__()
        self.m_inputSize = inputSize
        self.m_nDownSamples = nDownSamples

        N = 3  # the number of layer in each building block
        self.m_input = ConvInput(1, C, N-1, filterSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # inputSize = output

        self.m_downList, outputSize = self.addDownBBList(self.m_inputSize,C,C, 3, N)  # outputSize ={2*16*16}

        self.m_downList.append( DownBB(C, C, filter1st=(2,3,3), stride=(2,2,2), nLayers=N))      # outpusSize =(1*7*7}
        outputSize = self.getConvOutputTensorSize(outputSize,(2,3,3), (2,2,2), (0,0,0))
        self.m_bottleNeckSize = outputSize

        # for response prediction
        self.m_11Conv = BN_ReLU_Conv(C, 1, (1,1,1), (1,1,1), (0,0,0), False)           # outpusSize =(1*7*7}
        lenBn = 49
        self.m_fc11   = nn.Sequential(
                       nn.Linear(lenBn , lenBn//2),
                       nn.InstanceNorm1d(lenBn//2),
                       nn.ReLU(inplace=True),
                       nn.Linear(lenBn//2, lenBn//4),
                       nn.InstanceNorm1d(lenBn//4),
                       nn.ReLU(inplace=True),
                       nn.Linear(lenBn//4, Kr))

        # for segmentation reconstruction
        outputSize = self.getConvTransposeOutputTensorSize(outputSize, (2,3,3), (2,2,2), (0,0,0))
        self.m_upList, outputSize = self.addUpBBList(outputSize, C, C, 3, N)  # outputSize = 23*127*127
        self.m_upList.insert(0, UpBB(C, C, filter1st=(2,3,3), stride=(2,2,2), nLayers=N))

        self.m_upOutput = nn.Conv3d(C, Kup, (1,1,1), stride=(1,1,1))                   # outputSize = 23*127*127

    def encoderForward(self, inputx):
        x = self.m_input(inputx)
        for down in self.m_downList:
            x = down(x)
        # here x is the output at crossing point of sky watcher
        return x

    def responseForward(self, crossingx):
        # xr means x rightside output, or response output
        xr = crossingx
        xr = self.m_11Conv(xr)
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

