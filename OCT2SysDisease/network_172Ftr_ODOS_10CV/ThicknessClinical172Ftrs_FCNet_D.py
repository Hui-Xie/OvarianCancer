
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel
from framework.NetTools import  construct2DFeatureNet

# for input size: 9x9+7 and flat
class ThicknessClinical172Ftrs_FCNet_D(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.class01Percent[0] / hps.class01Percent[1]).to(hps.device)

        # where,
        # network structure:
        #  numThicknessFtr thickness========(FC)=======>(nThicknessLayer0) --|
        #  numClinicalFtr clinical  ======  (FC) ====>(numClinicalFtr)   --|==(FC_Widths....)=>1


        self.m_thicknessLayer0= nn.Sequential(
                    nn.Linear(hps.numThicknessFtr,hps.nThicknessLayer0),
                    nn.BatchNorm1d(hps.nThicknessLayer0),
                    nn.ReLU(),
                )
        self.m_clinicalLayer0 = nn.Sequential(
                    nn.Linear(hps.numClinicalFtr, hps.numClinicalFtr),
                    nn.BatchNorm1d(hps.numClinicalFtr),
                    nn.ReLU(),
                )

        # construct FC network after input layer 0
        self.m_linearLayerList = nn.ModuleList()
        widthInput = hps.nThicknessLayer0+ hps.numClinicalFtr
        nLayer = len(hps.fcWidths)
        for i, widthOutput in enumerate(hps.fcWidths):
            if i != nLayer - 1:
                layer = nn.Sequential(
                    nn.Linear(widthInput, widthOutput),
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
        thickness = x[:,0:self.hps.numThicknessFtr]
        clinical  = x[:,self.hps.numThicknessFtr:]
        x1 = self.m_thicknessLayer0(thickness)
        x2 = self.m_clinicalLayer0(clinical)
        x = torch.cat((x1,x2),dim=1)

        # FC layers after concatenated layer0
        for layer in self.m_linearLayerList:
            x = layer(x)

        x = x.squeeze(dim=-1)  # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



