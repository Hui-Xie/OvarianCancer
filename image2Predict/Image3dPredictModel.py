from framework.BasicModel import BasicModel
from framework.BuildingBlocks import *
import torch

# Predictive Model for treatment response

class Image3dPredictModel(BasicModel):
    def __init__(self, C,  K, inputSize, nDownSamples):
        super().__init__()
        self.m_inputSize = inputSize
        self.m_nDownSamples = nDownSamples
        # self.m_bottleNeckSize  = self.getDownSampleSize(self.m_inputSize, self.m_nDownSamples)
        lenBn = self.getProduct(self.m_bottleNeckSize)  # len of BottleNeck

        N = 3  # the number of layer in each building block
        self.m_input = ConvInput(1, C//2, N-1, filterSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # inputSize = output

        self.m_downList = nn.ModuleList()
        for i  in range(self.m_nDownSamples):
            if 0 == i:
                self.m_downList.append(DownBB(C//2, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N))
            else:
                self.m_downList.append(DownBB(C, C, filter1st=(3, 3, 3), stride=(2, 2, 2), nLayers=N))
        # for inputSize 147*281*281, after 6 downsamples, the output size is (1*3*3)

        self.m_fc11   = nn.Sequential(
                       nn.Linear(C*lenBn , C*lenBn//2),
                       nn.InstanceNorm1d(C*lenBn//2),
                       nn.ReLU(inplace=True),
                       nn.Linear(C*lenBn//2, C*lenBn//4),
                       nn.InstanceNorm1d(C*lenBn//4),
                       nn.ReLU(inplace=True),
                       nn.Linear(C*lenBn//4, K))

    def forward(self, inputx):
        x = self.m_input(inputx)
        for down in self.m_downList:
            x = down(x)
        x = torch.reshape(x, (1,x.numel()))
        x = self.m_fc11(x)
        return x
