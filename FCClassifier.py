#

from BasicModel import BasicModel
from ConvBlocks import LinearBlock
import torch.nn as nn
import torch


class FCClassifier(BasicModel):
    def __init__(self):
        super().__init__()

        self.m_layer0 = nn.LayerNorm(192, elementwise_affine=False)
        self.m_layer1 = LinearBlock(192, 100)
        self.m_layer2 = LinearBlock(100, 50)
        self.m_layer3 = LinearBlock(50, 25)
        self.m_layer4 = LinearBlock(25, 1)
        self.m_layer5 = nn.Sigmoid()

    def forward(self, x, gts=None):
        device = x.device
        x = self.m_layer0(x)
        x = self.m_layer1(x)
        x = self.m_layer2(x)
        x = self.m_layer3(x)
        x = self.m_layer4(x, useNonLinearActivation=False) # output size =1, do not use ReLU
        x = self.m_layer5(x)

        if gts is None:
            return x
        else:
            # compute loss (put loss here is to save main GPU memory)
            loss = torch.tensor(0.0).to(device)
            for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
                if weight == 0:
                    continue
                lossFunc.to(device)
                loss += lossFunc(x, gts) * weight

            return x, loss

