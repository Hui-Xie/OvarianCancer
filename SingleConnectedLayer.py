# Single Conneted Layer

import torch.nn as nn
import torch

class SingleConnectedLayer(nn.Module):
    def __init__(self, inFeatures):
        super().__init__()
        self.m_numFeatures = inFeatures
        F = self.m_numFeatures
        self.m_W0 = nn.Parameter(torch.zeros((F), dtype=torch.float, requires_grad=True))
        self.m_W1 = nn.Parameter(torch.zeros((F), dtype=torch.float, requires_grad=True))
        #self.m_W2 = nn.Parameter(torch.zeros((F), dtype=torch.float, requires_grad=True))
        # initialize W
        self.m_W1.data.fill_(0.01)
        #self.m_W2.data.fill_(0.01)
        #nn.init.normal_(self.m_W0, mean=0.0, std=1.0)
        #nn.init.normal_(self.m_W1, mean=0.0, std=1.0)
        #nn.init.normal_(self.m_W2, mean=0.0, std=1.0)


    def forward(self, x):
        """
        y = W0+W1*x for each feature
        """
        B,F = x.shape
        assert F == self.m_numFeatures
        F = self.m_numFeatures
        W0 = self.m_W0.expand((B,F)) # plane
        W1 = self.m_W1.diag()   # Diagonal matrix
        #W2 = self.m_W2.diag()   # Diagonal matrix
        #y = W0 + torch.mm(x,W1) + torch.mm(x**2,W2)  # linear regression output logits
        y = W0 + torch.mm(x, W1)
        return y


