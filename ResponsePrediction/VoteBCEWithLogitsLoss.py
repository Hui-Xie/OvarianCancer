
from torch.nn.modules.loss import _Loss
import torch

class VoteBCEWithLogitsLoss(_Loss):
    def __init__(self, pos_weight=1, weightedVote=False):
        super().__init__()
        self.m_posWeight = pos_weight
        self.m_weightedVote = weightedVote


    def forward(self, x, gts):
        B,F = x.shape
        gts =gts.reshape((B,1))
        sigmoidx = torch.sigmoid(x)
        gtsPlane = gts.expand((B,F))
        loss = -gtsPlane*torch.log(sigmoidx)*self.m_posWeight - (1.0-gtsPlane)*torch.log(1-sigmoidx)
        loss = loss.sum()*1.0/(B*F)
        if self.m_weightedVote:
            voteProb = sigmoidx.sum(dim=1)*1.0/F
        else:
            voteProb = (sigmoidx >=0.5).int().sum(dim=1) * 1.0 / F
        voteLogit = torch.log(voteProb / (1.0 - voteProb))
        return voteLogit, loss