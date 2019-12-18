#

from BasicModel import BasicModel
from SingleConnectedLayer import *
import torch.nn as nn
from ConvBlocks import LinearBlock
import torch


class VoteClassifier(BasicModel):
    def __init__(self):
        super().__init__()
        self.m_layers1 = nn.Sequential(
            nn.LayerNorm(192, elementwise_affine=False),
            #nn.BatchNorm1d(192),
            SingleConnectedLayer(192),
            nn.Tanh(),
            SingleConnectedLayer(192),
            nn.Tanh(),
            SingleConnectedLayer(192)
        )
        self.m_layers2 = nn.Sequential(
            nn.LayerNorm(192, elementwise_affine=False),
            # nn.BatchNorm1d(192, affine=False),
            LinearBlock(192, 120, normModule=nn.LayerNorm(120, elementwise_affine=False)),
            nn.Dropout(p=0.5),
            LinearBlock(120, 70, normModule=nn.LayerNorm(70, elementwise_affine=False)),
            nn.Dropout(p=0.5),
            LinearBlock(70, 40, normModule=nn.LayerNorm(40, elementwise_affine=False)),
            nn.Dropout(p=0.5),
            LinearBlock(40, 1, useNonLinearActivation=False)
            # output logits, which needs sigmoid inside the loss function.
        )

    def forward(self, x, gts=None):
        device = x.device
        x1 = self.m_layers1(x)
        x2 = self.m_layers2(x1)

        if gts is None:
            return x2  # output logits before sigmoid
        else:
            # compute loss (put loss here is to save main GPU memory)
            lossFunc0 = self.m_lossFuncList[0]
            weight0 = self.m_lossWeightList[0]
            _, loss0 = lossFunc0(x1, gts)* weight0

            lossFunc1 = self.m_lossFuncList[1]
            weight1 = self.m_lossWeightList[1]
            loss1 = lossFunc1(x2, gts) * weight1

            loss = (loss0 + loss1)/2.0
            return x2, loss

