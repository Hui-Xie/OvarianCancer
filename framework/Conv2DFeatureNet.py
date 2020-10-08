
import torch.nn as nn

import sys
sys.path.append("../..")
from framework.NetTools import construct2DFeatureNet


class Conv2DFeatureNet(nn.Module):
    def __init__(self, inputChannels, nStartFilters, nLayers, outputChannels,inputActivation):
        super().__init__()
        self.m_nLayers = nLayers
        self.m_downPoolings, self.m_downLayers = construct2DFeatureNet(inputChannels, nStartFilters, nLayers, inputActivation)

        C = nStartFilters * pow(2, nLayers-1)
        self.m_outputConv = nn.Sequential(
            nn.Conv2d(C, outputChannels, kernel_size=1, stride=1, padding=0, bias=True)
            # nn.BatchNorm2d(hps.outputChannels), #*** norm should not be before avgPooling ****
            # nn.Hardswish()
        )

    def forward(self,x):
        for i in range(self.m_nLayers):
            x = self.m_downPoolings[i](x)
            x = self.m_downLayers[i](x) + x
        x = self.m_outputConv(x)
        return x