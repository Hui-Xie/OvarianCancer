
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel

class ThicknessMap2HyperTensionNet_VGG16(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.hypertensionClassPercent[0] / hps.hypertensionClassPercent[1]).to(hps.device)

        self.m_conv1=nn.Sequential(
            nn.Conv2d(hps.inputChannels, hps.channels[0], kernel_size=(3,3), stride=(1,1), padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hps.channels[0], hps.channels[0], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=False),
        )
        self.m_pool1 = nn.MaxPool2d((2,2), stride=2)  # output size: C0x15x256

        self.m_conv2 = nn.Sequential(
            nn.Conv2d(hps.channels[0], hps.channels[1], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hps.channels[1], hps.channels[1], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=False),
        )
        self.m_pool2 = nn.MaxPool2d((2, 2), stride=2)  # output size: C1x7x128

        self.m_conv3 = nn.Sequential(
            nn.Conv2d(hps.channels[1], hps.channels[2], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hps.channels[2], hps.channels[2], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(hps.channels[2], hps.channels[2], kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.ReLU(inplace=False),
        )
        self.m_pool3 = nn.MaxPool2d((2, 2), stride=2)  # output size: C2x3x64

        self.m_conv4 = nn.Sequential(
            nn.Conv2d(hps.channels[2], hps.channels[3], kernel_size=(3, 3), stride=(1, 1), padding=0, bias=True),
            nn.ReLU(inplace=False), # output size: C3x1x62
            nn.Conv2d(hps.channels[3], hps.channels[3], kernel_size=(1, 9), stride=(1, 2), padding=0, bias=True),
            nn.ReLU(inplace=False),  # output size: C3x1x27
            nn.Conv2d(hps.channels[3], hps.channels[3], kernel_size=(1, 9), stride=(1, 2), padding=0, bias=True),
            nn.ReLU(inplace=False),  # output size: C3x1x10
        )
        self.m_pool4 = nn.AdaptiveMaxPool2d((1,1))  # output size: C3x1x1
        # here needs squeeze dim=-1 and dim =-2

        self.m_fc1 = nn.Sequential(
            nn.Linear(hps.channels[3], hps.channels[4]),
            nn.ReLU(inplace=False),
        )
        self.m_fc2 = nn.Linear(hps.channels[4], 1)

    def forward(self,x,t):
        x = self.m_conv1(x)
        x = self.m_pool1(x)
        x = self.m_conv2(x)
        x = self.m_pool2(x)
        x = self.m_conv3(x)
        x = self.m_pool3(x)
        x = self.m_conv4(x)
        x = self.m_pool4(x)
        x = x.squeeze(dim=-1)
        x = x.squeeze(dim=-1)
        x = self.m_fc1(x)
        x = self.m_fc2(x)
        x = x.squeeze(dim=-1)  # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



