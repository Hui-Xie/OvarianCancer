
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel

class ThicknessMap2HyperTensionNet_B(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.hypertensionClassPercent[0] / hps.hypertensionClassPercent[1]).to(hps.device)

        self.m_conv2DLayers = nn.Sequential(
            nn.Conv2d(hps.inputChannels, hps.channels[0], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hps.channels[0]),
            nn.Hardswish(), # 16x256
            nn.Dropout2d(p=hps.dropoutRates[0]),  # dropout after activation function
            nn.Conv2d(hps.channels[0], hps.channels[1], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hps.channels[1]),
            nn.Hardswish(),  # 8x128
            nn.Dropout2d(p=hps.dropoutRates[1]),
            nn.Conv2d(hps.channels[1], hps.channels[2], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hps.channels[2]),
            nn.Hardswish(),  # 4x64
            nn.Dropout2d(p=hps.dropoutRates[2]),
            nn.Conv2d(hps.channels[2], hps.channels[3], kernel_size=(4,3), stride=2, padding=(0,1), bias=True),
            nn.BatchNorm2d(hps.channels[3]),
            nn.Hardswish(),  # 1x32
            nn.Dropout2d(p=hps.dropoutRates[3])
        )
        # there follows a avgPool to average space features.
        self.m_adaptiveAvgPool1D = nn.AdaptiveAvgPool1d(1)

        self.m_fcLayers= nn.Linear(hps.channels[3], 1)


    def forward(self,x,t):
        x = self.m_conv2DLayers(x)  # BxChannels[3]x1x32
        x = x.squeeze(dim=-2)
        x = self.m_adaptiveAvgPool1D(x)  # BxChannels[3]x1
        x = x.squeeze(dim=-1)
        x = self.m_fcLayers(x)  # Bx1
        x = x.squeeze(dim=-1)   # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



