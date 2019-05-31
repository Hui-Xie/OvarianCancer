from BasicModel import BasicModel
from BuildingBlocks import *
import torch

# Predictive Model for treatment response

class Image3dPredictModel(BasicModel):
    def __init__(self, C,  K, inputSize, nDownSample):
        super().__init__()
        self.m_inputSize = inputSize
        self.m_nDownSample = nDownSample
        bottleNeckSize  = self.getBottleNeckSize()
        lenBn = self.getProduct(bottleNeckSize)  # len of BottleNeck

        # input size: C*D*H*W,  and output K class classification
        # C = 1024  # the number of channels after the first input layer.
        N = 4  # the number of layer in each building block
        self.m_input = ConvInput(1, C, N-1, filterSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # inputSize: 1*73*141*141; output:C*73*141*141

        self.m_down1 = DownBB(C, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output:C*36*70*70
        self.m_down2 = DownBB(C, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C*17*34*34
        self.m_down3 = DownBB(C, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C*8*16*16
        self.m_down4 = DownBB(C, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C*3*7*7
        self.m_down5 = DownBB(C, C,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C*1*3*3


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
        x = self.m_down1(x)
        x = self.m_down2(x)
        x = self.m_down3(x)
        x = self.m_down4(x)
        x = self.m_down5(x)
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
        return xSize

    @staticmethod
    def getProduct(aTuple):
        prod = 1
        for x in aTuple:
            prod *= x
        return prod
