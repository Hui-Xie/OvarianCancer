from SkyWatcherModel import SkyWatcherModel
from BuildingBlocks import *
import torch

# SkyWatcher Model 1, keep same number of filter in each layer in encoder and decoder

class SkyWatcherModel1 (SkyWatcherModel):
    def __init__(self, C,  Kr, Kup, inputSize):
        super().__init__()
        self.m_inputSize = inputSize

        N = 3  # the number of layer in each building block
        self.m_input = ConvInput(1, C, N, filterSize=(3, 3, 3), stride=(1, 1, 1),
                                 padding=(1, 1, 1))  # inputSize = output

        self.m_downList, outputSize = self.addDownBBList(self.m_inputSize, C, C, 3, N)  # outputSize ={2*16*16}

        self.m_downList.append(DownBB(C, C, filter1st=(2, 3, 3), stride=(2, 2, 2), nLayers=N))  # outpusSize =(1*7*7}
        outputSize = self.getConvOutputTensorSize(outputSize, (2, 3, 3), (2, 2, 2), (0, 0, 0))
        self.m_bottleNeckSize = (C,) + outputSize

        # for response prediction
        self.m_11Conv = BN_ReLU_Conv(C, 1, (1, 1, 1), (1, 1, 1), (0, 0, 0), False)  # outpusSize =(1*7*7}
        lenBn = 49
        self.m_fc11 = nn.Sequential(
            nn.Linear(lenBn, lenBn // 2),
            nn.ReLU(inplace=True),
            nn.Linear(lenBn // 2, lenBn // 4),
            nn.ReLU(inplace=True),
            nn.Linear(lenBn // 4, Kr))

        # for segmentation reconstruction
        outputSize = self.getConvTransposeOutputTensorSize(outputSize, (2, 3, 3), (2, 2, 2), (0, 0, 0))
        self.m_upList, outputSize = self.addUpBBList(outputSize, C, C, 3, N)  # outputSize = 23*127*127
        self.m_upList.insert(0, UpBB(C, C, filter1st=(2, 3, 3), stride=(2, 2, 2), nLayers=N))

        self.m_upOutput = nn.Conv3d(C, Kup, (1, 1, 1), stride=(1, 1, 1))  # outputSize = 23*127*127

