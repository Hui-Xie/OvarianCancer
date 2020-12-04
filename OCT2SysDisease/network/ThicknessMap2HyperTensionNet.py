
import torch
import torch.nn as nn
from framework.BasicModel import  BasicModel

class ThicknessMap2HyperTensionNet(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.hypertensionClassPercent[0] / hps.hypertensionClassPercent[1]).to(hps.device)

        self.m_conv2DLayers = nn.Sequential(
            nn.Conv2d(hps.inputChannels, hps.channels[0], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hps.channels[0]),
            nn.Hardswish(), # 16x256
            nn.Conv2d(hps.channels[0], hps.channels[1], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hps.channels[1]),
            nn.Hardswish(),  # 8x128
            nn.Conv2d(hps.channels[1], hps.channels[2], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(hps.channels[2]),
            nn.Hardswish(),  # 4x64
            nn.Conv2d(hps.channels[2], hps.channels[3], kernel_size=(4,3), stride=2, padding=(0,1), bias=True),
            nn.BatchNorm2d(hps.channels[3]),
            nn.Hardswish()  # 1x32
        )
        self.m_conv1DLayers = nn.Sequential(
            nn.Conv1d(hps.channels[3], hps.channels[4], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(hps.channels[4]),
            nn.Hardswish(),  # 16
            nn.Conv1d(hps.channels[4], hps.channels[5], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(hps.channels[5]),
            nn.Hardswish(),  # 8
            nn.Conv1d(hps.channels[5], hps.channels[6], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm1d(hps.channels[6]),
            nn.Hardswish(),  # 4
            nn.Conv1d(hps.channels[6], hps.channels[7], kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(hps.channels[7]),
            nn.Hardswish()  # 1
        )
        self.m_fcLayers= nn.Linear(hps.channels[7], 1)


    def forward(self,x,t):
        x = self.m_conv2DLayers(x)
        x = x.squeeze(dim=-2)
        x = self.m_conv1DLayers(x)
        x = x.squeeze(dim=-1)
        x = self.m_fcLayers(x)
        x = x.squeeze(dim=-1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



