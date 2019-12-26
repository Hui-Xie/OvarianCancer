#

from framework.BasicModel import BasicModel
from framework.SingleConnectedLayer import *
import torch.nn as nn
from framework.ConvBlocks import LinearBlock
import torch

preTrainEpochs = 10000

class VoteClassifier(BasicModel):
    def __init__(self):
        super().__init__()
        self.m_layers1 = nn.Sequential(
            nn.LayerNorm(192, elementwise_affine=False),
            #nn.BatchNorm1d(192),
            SingleConnectedLayer(192),
            nn.ReLU(),
            SingleConnectedLayer(192),
            nn.ReLU(),
            SingleConnectedLayer(192)   #output: B*F feature logits
        )
        self.m_layers2 = nn.Sequential(
            #nn.LayerNorm(192, elementwise_affine=False),
            nn.Sigmoid(),
            # nn.BatchNorm1d(192, affine=False),
            LinearBlock(192, 120, normModule=nn.LayerNorm(120, elementwise_affine=False)),
            nn.Dropout(p=0.5),
            LinearBlock(120, 70, normModule=nn.LayerNorm(70, elementwise_affine=False)),
            nn.Dropout(p=0.3),
            LinearBlock(70, 40, normModule=nn.LayerNorm(40, elementwise_affine=False)),
            LinearBlock(40, 1, useNonLinearActivation=False)
            # output logits, which needs sigmoid inside the loss function.
        )

    def forward(self, x, gts):
        # this is for 75% max test accuracy network:  20191219_123556:

        device = x.device
        loss0 = torch.tensor(0.0).to(device)
        loss1 = torch.tensor(0.0).to(device)

        x0 = self.m_layers1(x)
        lossFunc0 = self.m_lossFuncList[0]
        weight0 = self.m_lossWeightList[0]
        voteLogit, loss0 = lossFunc0(x0, gts)
        loss0 = loss0*weight0

        if self.training:
            if self.m_epoch > preTrainEpochs:
                x1 = self.m_layers2(x0)
                lossFunc1 = self.m_lossFuncList[1]
                weight1 = self.m_lossWeightList[1]
                gts = gts.view_as(x1)
                loss1 = lossFunc1(x1, gts) * weight1
                loss = loss0 + loss1
            else:
                x1 = voteLogit
                loss = loss0
        else:
            x1 = self.m_layers2(x0)
            lossFunc1 = self.m_lossFuncList[1]
            weight1 = self.m_lossWeightList[1]
            gts = gts.view_as(x1)
            loss1 = lossFunc1(x1, gts) * weight1
            loss = loss0 + loss1

        return x1, loss





