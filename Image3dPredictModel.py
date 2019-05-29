from BasicModel import BasicModel
from BuildingBlocks import *
import torch

# Predictive Model for treatment response

class Image3dPredictModel(BasicModel):
    def __init__(self, C,  K):
        super().__init__()
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
                       nn.Linear(C*9 , C*4),
                       nn.InstanceNorm1d(C*4),
                       nn.ReLU(inplace=True),
                       nn.Linear(C*4, C),
                       nn.InstanceNorm1d(C),
                       nn.ReLU(inplace=True),
                       nn.Linear(C, K))

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