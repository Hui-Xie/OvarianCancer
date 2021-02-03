
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

        self.m_thicknessLayer0= nn.Sequential(
                    nn.Linear(hps.numThicknessFtr,10),
                    nn.BatchNorm1d(10),
                    nn.ReLU(),
                )
        self.m_clinicalLayer0 = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),  # 1*1 conv to adjust clinical parameter.
                    nn.BatchNorm1d(1),  ## normorlization an batch dimension.
                    nn.ReLU(),
                )

        self.m_thicknessClinicalLayer= nn.Sequential(
                    nn.Linear(20,1),
                )

    def forward(self,x,t):
        thickness = x[:,0:self.hps.numThicknessFtr]
        clinical  = x[:,self.hps.numThicknessFtr:].unsqueeze(dim=1)
        x1 = self.m_thicknessLayer0(thickness)
        x2 = self.m_clinicalLayer0(clinical).squeeze(dim=1)
        x12 = torch.cat((x1,x2),dim=1)
        x = self.m_thicknessClinicalLayer(x12)

        x = x.squeeze(dim=-1)  # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



