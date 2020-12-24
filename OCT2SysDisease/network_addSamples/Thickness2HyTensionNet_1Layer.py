
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel
# for input size: 9x15x12

class Thickness2HyTensionNet_1Layer(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.hypertensionClassPercent[0] / hps.hypertensionClassPercent[1]).to(hps.device)

        self.m_conv2DLayers = nn.Sequential(
            nn.Conv2d(hps.inputChannels, hps.channels[0], kernel_size=(15,12), stride=(1,1), padding=0, bias=True),
            nn.ReLU(inplace=False), # BxCx1x1
        )
        self.m_dropout = nn.Dropout(p=hps.dropoutRate, inplace=False)  # dropout after activation function
        self.m_fcLayers= nn.Linear(hps.channels[0], 1)


    def forward(self,x,t):
        x = self.m_conv2DLayers(x)  # BxC[0]x1x1
        x = x.squeeze(dim=-2)
        x = x.squeeze(dim=-1)  # BxC[0]
        x = self.m_dropout(x)
        x = self.m_fcLayers(x)  # Bx1
        x = x.squeeze(dim=-1)   # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



