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
        self.m_input = ConvInput(1, C, N, filterSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # inputSize: 1*147*281*281; output:C*147*281*281

        self.m_down1 = DownBB(C,      C // 2,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output:C//2*73*140*140
        self.m_down2 = DownBB(C // 2, C // 4,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C//4*36*69*69
        self.m_down3 = DownBB(C // 4, C // 8,   filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C//8*17*34*34
        self.m_down4 = DownBB(C // 8, C // 16,  filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C//16*8*16*16
        self.m_down5 = DownBB(C // 16, C // 32, filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C//32*3*7*7
        self.m_down6 = DownBB(C // 32, C // 32, filter1st = (3, 3, 3), stride=(2, 2, 2), nLayers=N)          # output: C//32*1*3*3

        self.m_fc11   = nn.Sequential(
                       nn.Linear(C//32 *9 , C//32 * 3),
                       nn.BatchNorm1d(C//32 *3),
                       nn.ReLU(inplace=True),
                       nn.Linear(C//32*3, C//32),
                       nn.BatchNorm1d(C//32),
                       nn.ReLU(inplace=True),
                       nn.Linear(C//32, K))

    def forward(self, inputx):
        x = self.m_input(inputx)
        x = self.m_down1(x)
        x = self.m_down2(x)
        x = self.m_down3(x)
        x = self.m_down4(x)
        x = self.m_down5(x)
        x = self.m_down6(x)
        x = torch.reshape(x, (x.numel(), 1))
        x = self.m_fc11(x)
        return x