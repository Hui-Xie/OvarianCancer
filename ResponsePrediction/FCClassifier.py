#

from BasicModel import BasicModel
from ConvBlocks import LinearBlock
import torch.nn as nn
import torch


class FCClassifier(BasicModel):
    def __init__(self):
        super().__init__()

        self.m_layers = nn.Sequential(
            nn.LayerNorm(192, elementwise_affine=False),
            #nn.BatchNorm1d(192, affine=False),
            LinearBlock(192, 120, normModule=nn.LayerNorm(120, elementwise_affine=False)),
            nn.Dropout(p=0.5),
            LinearBlock(120, 70, normModule=nn.LayerNorm(70, elementwise_affine=False)),
            nn.Dropout(p=0.5),
            LinearBlock(70, 40, normModule=nn.LayerNorm(40, elementwise_affine=False)),
            nn.Dropout(p=0.5),
            LinearBlock(40, 1, useNonLinearActivation=False)  # output logits, which needs sigmoid inside the loss function.
        )


    def forward(self, x, gts=None):
        device = x.device
        x = self.m_layers(x)

        if gts is None:
            return x  # output logits
        else:
            # compute loss (put loss here is to save main GPU memory)
            loss = torch.tensor(0.0).to(device)
            for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
                if weight == 0:
                    continue
                lossFunc.to(device)
                gts = gts.view_as(x)
                loss += lossFunc(x, gts) * weight

            return x, loss

