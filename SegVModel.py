import torch.nn as nn
import torch
import torch.nn.functional as F


class SegVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m_dropoutProb = 0.2
        self.m_dropout3d = nn.Dropout3d(p=self.m_dropoutProb, inplace=True)
        self.m_dropout2d = nn.Dropout2d(p=self.m_dropoutProb, inplace=True)
        self.m_optimizer = None
        self.m_lossFuncList = []
        self.m_lossWeightList = []

    def forward(self, x):
        pass

    def setOptimizer(self, optimizer):
        self.m_optimizer = optimizer

    def appendLossFunc(self, lossFunc, weight = 1.0):
        self.m_lossFuncList.append(lossFunc)
        self.m_lossWeightList.append(weight)

    def lossFunctionsInfo(self):
        return f'Loss Functions List: ' + f'\t'.join(f'{type(loss).__name__} with weight of {weight}; ' for loss, weight in zip(self.m_lossFuncList, self.m_lossWeightList))

    def updateLossWeightList(self, weightList):
        self.m_lossWeightList = weightList

    def getLossWeightList(self):
        return self.m_lossWeightList

    def batchTrain(self, inputs, labels):
        self.m_optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = torch.tensor(0.0).cuda()
        for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
            if weight == 0:
                continue
            loss += lossFunc(outputs,labels)*weight
        loss.backward()
        self.m_optimizer.step()
        return loss.item()

    def batchTrainMixup(self, inputs, labels1, labels2, lambdaInBeta):
        self.m_optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = torch.tensor(0.0).cuda()
        for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
            if weight == 0:
                continue
            if lambdaInBeta != 0:
                loss += lossFunc(outputs,labels1)*weight*lambdaInBeta
            if 1-lambdaInBeta != 0:
                loss += lossFunc(outputs,labels2)*weight*(1-lambdaInBeta)
        loss.backward()
        self.m_optimizer.step()
        return loss.item()

    def batchTest(self, inputs, labels):
        outputs = self.forward(inputs)
        loss = torch.tensor(0.0).cuda()
        for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
            if weight == 0:
                continue
            loss += lossFunc(outputs, labels) * weight
        return loss.item(), outputs

    def getParametersScale(self):
        sumPara = 0
        params = self.parameters()
        for param in params:
            sumPara += param.nelement()
        return f"Network has total {sumPara} parameters."

    def setDropoutProb(self, prob):
        self.m_dropoutProb = prob
        return f"Info: network dropout rate = {self.m_dropoutProb}"


