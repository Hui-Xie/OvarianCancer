from BasicModel import BasicModel
from ModuleBuildingBlocks import *
import torch

# Predictive Model for treatment response

class PredictModel(BasicModel):
    def __init__(self, C, K):
        super().__init__()
        # input size: C*H*W,  and output K class classification

        N = 4  # the number of layer in each building block
        self.m_input = ConvInput(C, C//2, N)                             # inputSize: C*51*49; output:C//2*51*49

        self.m_down1 = Down2dBB(C//2, C//4, (3, 5), stride=(2, 2), nLayers=N)  # output:C//4*25*23
        self.m_down2 = Down2dBB(C//4, C//8, (5, 3), stride=(2, 2), nLayers=N)  # output: C//8*11*11
        self.m_down3 = Down2dBB(C//8, C//16, (3, 3), stride=(2, 2), nLayers=N)  # output: C//16*5*5
        self.m_down4 = Down2dBB(C//16, C//32, (3, 3), stride=(2, 2), nLayers=N)  # output: C//32*2*2
        self.m_down5 = BN_ReLU_Conv2d(C//32, C//32, filterSize=(2,2), stride=(1,1), padding=(0,0), order=False)   # output: C//32*1*1
        self.m_fc11   = nn.Sequential(
                       nn.Linear(C//32, C//64),
                       nn.BatchNorm1d(C//64),
                       nn.ReLU(inplace=True),
                       nn.Linear(C//64, C//64),
                       nn.BatchNorm1d(C//64),
                       nn.ReLU(inplace=True),
                       nn.Linear(C//64, K))

    def forward(self, inputx):
        x = self.m_input(inputx)
        x = self.m_down1(x)
        x = self.m_down2(x)
        x = self.m_down3(x)
        x = self.m_down4(x)
        x = self.m_down5(x)
        x = torch.squeeze(x)
        x = self.m_fc11(x)
        return x
