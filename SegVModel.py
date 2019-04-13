import torch.nn as nn


class SegVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_dropoutProb = 0.2
        self.m_dropout3d = nn.Dropout3d(p=self.m_dropoutProb)
        self.m_dropout2d = nn.Dropout2d(p=self.m_dropoutProb)

    def setOptimizer(self, optimizer):
        self.m_optimizer = optimizer

    def setLossFunc(self, lossFunc):
        self.m_lossFunc = lossFunc

    def batchTrain(self, inputs, labels):
        self.m_optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.m_lossFunc(outputs, labels)
        loss.backward()
        self.m_optimizer.step()
        return loss.item()

    def batchTest(self, inputs, labels):
        outputs = self.forward(inputs)
        loss = self.m_lossFunc(outputs, labels)
        return loss.item(), outputs

    def printParametersScale(self):
        sum = 0
        params = self.parameters()
        for param in params:
            sum += param.nelement()
        print(f"Network has total {sum} parameters.")

    def setDropoutProb(self, prob):
        self.m_dropoutProb = prob
        print(f"Info: network dropout rate = {self.m_dropoutProb}")
