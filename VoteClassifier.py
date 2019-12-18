#

from BasicModel import BasicModel
from SingleConnectedLayer import *
import torch.nn as nn
import torch


class VoteClassifier(BasicModel):
    def __init__(self):
        super().__init__()


        self.m_layers = nn.Sequential(
            #nn.LayerNorm(192, elementwise_affine=False),
            nn.BatchNorm1d(192),
            SingleConnectedLayer(192)

        )


    def forward(self, x, gts=None):
        device = x.device
        x = self.m_layers(x)

        if gts is None:
            return x  # output logits before sigmoid
        else:
            # compute loss (put loss here is to save main GPU memory)
            lossFunc = self.getOnlyLossFunc()
            predictProb, loss = lossFunc(x, gts)
            return predictProb, loss

