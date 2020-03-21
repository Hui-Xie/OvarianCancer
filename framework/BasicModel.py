import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
from framework.BuildingBlocks import *


class BasicModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.m_dropoutProb = 0.5
        # self.m_dropout3d = nn.Dropout3d(p=self.m_dropoutProb)
        # self.m_dropout2d = nn.Dropout2d(p=self.m_dropoutProb)
        # self.m_dropout1d = nn.Dropout(p=self.m_dropoutProb)
        self.m_optimizer = None
        self.m_lossFuncList = []
        self.m_lossWeightList = []
        self.m_consistencyLoss = None
        self.m_lossEpochList = []  # specify the running epochs for each loss
        self.m_epoch = 0
        self.m_currentLossFunc = None
        self.m_runParametersDict={}  # it will arbitrary parameters of network including epoch, loss etc.

    def forward(self, x):
        pass

    def updateRunParameter(self, nameStr, value):
        self.m_runParametersDict[nameStr] = value

    def getRunParameter(self, nameStr):
        if nameStr in self.m_runParametersDict:
            return self.m_runParametersDict[nameStr]
        else:
            return False

    def setOptimizer(self, optimizer):
        self.m_optimizer = optimizer

    def appendLossFunc(self, lossFunc, weight=1.0, epochs=1000):
        self.m_lossFuncList.append(lossFunc)
        self.m_lossWeightList.append(weight)
        self.m_lossEpochList.append(epochs)

    def getOnlyLossFunc(self):
        assert (1 == len(self.m_lossFuncList))
        return self.m_lossFuncList[0]

    def lossFunctionsInfo(self):
        return f'Loss Functions List: ' + f'\t'.join(f'{type(loss).__name__} with weight of {weight}; ' for loss, weight in zip(self.m_lossFuncList, self.m_lossWeightList))

    def getCurrentLossFunc(self):
        N = len(self.m_lossFuncList)
        accumulateEpochs = self.m_lossEpochList[0]
        if self.training:
            for i in range(N):
                if self.m_epoch < accumulateEpochs:
                    self.m_currentLossFunc = self.m_lossFuncList[i]
                    return self.m_currentLossFunc, self.m_lossWeightList[i]
                else:
                    if i+1 <= N-1:
                        accumulateEpochs +=self.m_lossEpochList[i+1]
                    continue
        self.m_currentLossFunc = self.m_lossFuncList[N-1]
        return self.m_currentLossFunc, self.m_lossWeightList[N-1]


    def updateLossWeightList(self, weightList):
        self.m_lossWeightList = weightList

    def getLossWeightList(self):
        return self.m_lossWeightList

    def getLossEpochList(self):
        return self.m_lossEpochList

    def batchTrain(self, inputs, labels):
        self.m_optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = torch.tensor(0.0).to(inputs.device)
        for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
            if weight == 0:
                continue
            loss += lossFunc(outputs,labels)*weight
        loss.backward()
        self.m_optimizer.step()
        return outputs, loss.item()

    def batchTrainMixup(self, inputs, labels1, labels2, lambdaInBeta):
        self.m_optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = torch.tensor(0.0).to(inputs.device)
        for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
            if weight == 0:
                continue
            if lambdaInBeta != 0:
                loss += lossFunc(outputs,labels1)*weight*lambdaInBeta
            if 1-lambdaInBeta != 0:
                loss += lossFunc(outputs,labels2)*weight*(1-lambdaInBeta)
        loss.backward()
        self.m_optimizer.step()
        return outputs, loss.item()

    def batchTest(self, inputs, labels):
        outputs = self.forward(inputs)
        loss = torch.tensor(0.0).to(inputs.device)
        for lossFunc, weight in zip(self.m_lossFuncList, self.m_lossWeightList):
            if weight == 0:
                continue
            loss += lossFunc(outputs, labels) * weight
        return outputs, loss.item()

    def getParametersScale(self):
        sumPara = 0
        params = self.parameters()
        for param in params:
            sumPara += param.nelement()
        return f"Network has total {sumPara:,d} parameters."

    def setDropoutProb(self, prob):
        self.m_dropoutProb = prob
        self.m_dropout2d.p = prob
        self.m_dropout3d.p = prob
        return f"Info: network dropout rate = {self.m_dropoutProb}"

    def getLR(self):
        return float(self.m_optimizer.param_groups[0]['lr'])

    @staticmethod
    def getConvOutputTensorSize(inputSize, filter, stride, padding):
        dim = len(inputSize)
        xSize = list(inputSize)
        for i in range(dim):
            xSize[i] = (xSize[i] + 2*padding[i]- filter[i]) // stride[i] + 1
        xSize = tuple(xSize)
        return xSize

    @staticmethod
    def getConvTransposeOutputTensorSize(inputSize, filter, stride, padding):
        dim = len(inputSize)
        xSize = list(inputSize)
        for i in range(dim):
            xSize[i] = (xSize[i] - 1)*stride[i] - 2* padding[i]+filter[i]
        xSize = tuple(xSize)
        return xSize

    @staticmethod
    def getProduct(aTuple):
        prod = 1
        for x in aTuple:
            prod *= x
        return prod

    @staticmethod
    def isTensorSizeLessThan(tensorSize, value):
        for x in tensorSize:
            if x < value:
                 return True
        return False

    @staticmethod
    def addDownBBList(inputSize, Cin, Cout, nDownSamples, nInBB):
        downList = nn.ModuleList()
        outputSize = inputSize
        dim = len(inputSize)
        filter = (3,) * dim
        stride = (2,) * dim
        padding = (0,) * dim
        for i in range(nDownSamples):
            if 0 == i:
                downList.append(DownBB(Cin, Cout, filter1st=filter, stride=stride, nLayers=nInBB))
            else:
                downList.append(DownBB(Cout, Cout, filter1st=filter, stride=stride, nLayers=nInBB))

            outputSize = BasicModel.getConvOutputTensorSize(outputSize, filter, stride, padding)
            if BasicModel.isTensorSizeLessThan(outputSize, 3):
                print(f"Warning: at the {i}th downSample with inputSize = {inputSize}, the outputSize = {outputSize}  has elements less than 3.")
                break

        return downList, outputSize

    @staticmethod
    def addUpBBList(inputSize, Cin, Cout, nUpSamples, nInBB):
        downList = nn.ModuleList()
        outputSize = inputSize
        dim = len(inputSize)
        filter = (3,) * dim
        stride = (2,) * dim
        padding = (0,) * dim
        for i in range(nUpSamples):
            if 0 == i:
                downList.append(UpBB(Cin, Cout, filter1st=filter, stride=stride, nLayers=nInBB))
            else:
                downList.append(UpBB(Cout, Cout, filter1st=filter, stride=stride, nLayers=nInBB))

            outputSize = BasicModel.getConvTransposeOutputTensorSize(outputSize, filter, stride, padding)
        return downList, outputSize

    @staticmethod
    def addDownBBListWithMoreFilters(inputSize, Cin, nDownSamples, nInBB):
        """

        :param inputSize:
        :param Cin:
        :param nDownSamples:
        :param nInBB:
        :return: downList, outputSize, C at the final layer.
        """
        downList = nn.ModuleList()
        outputSize = inputSize
        dim = len(inputSize)
        filter = (3,) * dim
        stride = (2,) * dim
        padding = (0,) * dim
        C = Cin   # channels number
        for i in range(nDownSamples):
            downList.append(DownBB(C,  2*C, filter1st=filter, stride=stride, nLayers=nInBB))
            outputSize = BasicModel.getConvOutputTensorSize(outputSize, filter, stride, padding)
            C = 2*C
            if BasicModel.isTensorSizeLessThan(outputSize, 3):
                print(
                    f"Warning: at the {i}th downSample with inputSize = {inputSize}, the outputSize = {outputSize}  has elements less than 3.")
                break

        return downList, outputSize, C

    @staticmethod
    def addUpBBListWithLessFilters(inputSize, Cin, nUpSamples, nInBB):
        """

        :param inputSize:
        :param Cin:
        :param nUpSamples:
        :param nInBB:
        :return: downList, outputSize, C at the final layer
        """
        downList = nn.ModuleList()
        outputSize = inputSize
        dim = len(inputSize)
        filter = (3,) * dim
        stride = (2,) * dim
        padding = (0,) * dim
        C = Cin
        for i in range(nUpSamples):
            downList.append(UpBB(C, C//2, filter1st=filter, stride=stride, nLayers=nInBB))
            outputSize = BasicModel.getConvTransposeOutputTensorSize(outputSize, filter, stride, padding)
            C = C//2
        return downList, outputSize, C

    @staticmethod
    def initializeWeights(m):
        """
        copy from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5 at June 6th, 2019
        :param m:  model.
        :return:
        """
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)

        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)

        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)

        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)

        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)

        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)

        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)

        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)

        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)

        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            # init.normal_(m.bias.data)
            init.constant_(m.bias.data, 0)

        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        else:
            #print(f"{m.__class__.__name__} does not support initialization in initializeWeights function.")
            pass


