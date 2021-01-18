
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel
from framework.NetTools import  construct2DFeatureNet

# for input size: 9x31x25
class Thickness9x9_FCNet(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.class01Percent[0] / hps.class01Percent[1]).to(hps.device)

        self.m_downPoolings, self.m_downLayers = construct2DFeatureNet(hps.inputChannels, hps.channels, hps.nLayers, numConvEachLayer=hps.numConvEachLayer)
        #output size: Bxchannel[nlayers-1]x3x3

        # use 1x1 conv to avoid conv4 to fully connected layer.
        self.m_conv4 = nn.Sequential(
            nn.Conv2d(hps.channels[hps.nLayers-1], hps.channels[hps.nLayers], kernel_size=(1, 1), stride=(1, 1), padding=0, bias=True),
            nn.ReLU(inplace=False),  # output size: BxC3x3x3
        )
        # remember: Before the FC layer, do not use batch Norm.
        # as normalization will kill same feature along normalization dimension.
        self.m_pool4 = nn.AdaptiveAvgPool2d((1, 1))  # output size: BxC3x1x1

        self.m_dropout= nn.Dropout(p=hps.dropoutRate, inplace=False)  # dropout after activation function

        self.m_fc = nn.Linear(hps.channels[hps.nLayers], 1)

    def forward(self,x,t):

        # down path of Unet
        for i in range(self.hps.nLayers):
            x = self.m_downPoolings[i](x)
            x = self.m_downLayers[i](x) + x

        x = self.m_conv4(x)
        x = self.m_pool4(x)
        x = x.squeeze(dim=-1)
        x = x.squeeze(dim=-1)
        x = self.m_dropout(x)
        x = self.m_fc(x)
        x = x.squeeze(dim=-1)  # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



