
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel

class ThicknessMap2HyperTensionNet_D(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.hypertensionClassPercent[0] / hps.hypertensionClassPercent[1]).to(hps.device)

        self.m_conv2DLayers = nn.Sequential(
            nn.Conv2d(hps.inputChannels, hps.channels[0], kernel_size=(31,31), stride=(1,5), padding=0, bias=True),
            nn.BatchNorm2d(hps.channels[0]),
            nn.ReLU(inplace=False), # 1x97
            nn.Dropout2d(p=hps.dropoutRates[0], inplace=False),  # dropout after activation function
            # this Dropout2D has same affect of dropout before Linear layer.
        )
        # there follows an avgPool to average space features.
        self.m_adaptiveAvgPool1D = nn.AdaptiveAvgPool1d(1)

        self.m_fcLayers= nn.Linear(hps.channels[0], 1)


    def forward(self,x,t):
        x = self.m_conv2DLayers(x)  # BxChannels[1]x1x97
        x = x.squeeze(dim=-2)
        x = self.m_adaptiveAvgPool1D(x)  # BxChannels[1]x1
        x = x.squeeze(dim=-1)
        x = self.m_fcLayers(x)  # Bx1
        x = x.squeeze(dim=-1)   # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



