from BasicModel import BasicModel
from BuildingBlocks import *
import torch

# Predictive Model for treatment response

class Image3dPredictModel(BasicModel):
    def __init__(self, C,  K, inputSize, nDownSample):
        super().__init__()
        self.m_inputSize = inputSize
        self.m_nDownSample = nDownSample
        self.m_bottleNeckSize  = self.getBottleNeckSize()
        lenBn = self.getProduct(self.m_bottleNeckSize)  # len of BottleNeck

        N = 4  # the number of layer in each building block
        self.m_input = ConvInput(1, C, N-1, filterSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # inputSize = output

        self.m_downList = nn.ModuleList()
        for _  in range(self.m_nDownSample):
            self.m_downList.append(DownBB(C, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N))

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

    def getBottleNeckSize(self):
        dim = len(self.m_inputSize)
        xSize = list(self.m_inputSize)
        for _ in range(self.m_nDownSample):
            for i in range(dim):
                xSize[i] = (xSize[i]-3)//2 +1
        xSize = tuple(xSize)
        print(f"the output size of bottle neck layer : {xSize}")
        return xSize

    @staticmethod
    def getProduct(aTuple):
        prod = 1
        for x in aTuple:
            prod *= x
        return prod
