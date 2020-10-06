
import torch.nn as nn

import sys
sys.path.append("../..")
from framework.NetTools import construct2DFeatureNet


class Conv2DFeatureNet(nn.Module):
    def __init__(self, inputChannels, nStartFilters, nLayers):
        super().__init__()
        self.m_nLayers = nLayers
        self.m_downPoolings, self.m_downLayers = construct2DFeatureNet(inputChannels, nStartFilters, nLayers)

    def forward(self,x):
        for i in range(self.m_nLayers):
            x = self.m_downPoolings[i](x)
            x = self.m_downLayers[i](x) + x
        return x