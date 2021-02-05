
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

        # where,
        # network structure:
        #  numThicknessFtr thickness========(FC)=======>(nThicknessLayer0) --|
        #  numClinicalFtr clinical  ======(1x1Conv)====>(numClinicalFtr)   --|==(FC_Widths....)=>1
        # parameters: 20*10 + 10*2 + 21*20 +21x1 =  661

        self.m_thicknessLayer0= nn.Sequential(
                    nn.Linear(hps.numThicknessFtr,hps.nThicknessLayer0),
                    nn.BatchNorm1d(hps.nThicknessLayer0),
                    nn.ReLU(),
                )
        self.m_clinicalLayer0 = nn.Sequential(
                    nn.Conv1d(1, 1, kernel_size=1, stride=1, padding=0),  # 1*1 conv to adjust clinical parameter.
                    nn.BatchNorm1d(1),  ## normorlization an batch dimension.
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
        clinical  = x[:,self.hps.numThicknessFtr:].unsqueeze(dim=1)
        x1 = self.m_thicknessLayer0(thickness)
        x2 = self.m_clinicalLayer0(clinical).squeeze(dim=1)
        x = torch.cat((x1,x2),dim=1)

        # FC layers after concatenated layer0
        for layer in self.m_linearLayerList:
            x = layer(x)

        x = x.squeeze(dim=-1)  # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss



