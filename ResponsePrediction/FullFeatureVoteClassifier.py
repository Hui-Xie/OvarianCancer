# full feature Vote Classifier

from framework.BasicModel import BasicModel
from framework.SingleConnectedLayer import *
import torch.nn as nn
from framework.ConvBlocks import LinearBlock
import torch

class FullFeatureVoteClassifier(BasicModel):
    def __init__(self, numFeatures):
        super().__init__()
        self.m_layers1 = nn.Sequential(
            SingleConnectedLayer(numFeatures)
        )      #output: B*F feature logits

    def forward(self, x, gts):
        device = x.device

        x0 = self.m_layers1(x)
        lossFunc0 = self.m_lossFuncList[0]
        weight0 = self.m_lossWeightList[0]
        voteLogit, loss = lossFunc0(x0, gts)
        loss = loss*weight0

        return voteLogit, loss





