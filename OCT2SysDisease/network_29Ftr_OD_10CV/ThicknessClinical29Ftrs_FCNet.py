
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel
from framework.NetTools import  construct2DFeatureNet

# for input size: 9x9+7 and flat
class ThicknessClinical29Ftrs_FCNet(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.class01Percent[0] / hps.class01Percent[1]).to(hps.device)

        # construct FC network
        self.m_linearLayerList = nn.ModuleList()

        widthInput = hps.inputWidth
        nLayer = len(hps.fcWidths)
        for i, widthOutput in enumerate(hps.fcWidths):
            if i != nLayer-1:
                layer = nn.Sequential(
                    nn.Linear(widthInput,widthOutput),
                    nn.BatchNorm1d(widthOutput),
                    nn.ReLU(),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(widthInput, widthOutput),
                )
            self.m_linearLayerList.append(layer)
            widthInput = widthOutput



    def forward(self,x,t):
        # FC layers with batchnorm1d
        for layer in self.m_linearLayerList:
            x = layer(x)

        x = x.squeeze(dim=-1)  # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



