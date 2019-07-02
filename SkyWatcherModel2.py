from SkyWatcherModel import SkyWatcherModel
from BuildingBlocks import *
import torch

# SkyWatcher Model 2, double the number of filers along deep layer in the encoder

class SkyWatcherModel2 (SkyWatcherModel):
    def __init__(self, C,  Kr, Kup, inputSize):
        super().__init__()
        self.m_inputSize = inputSize

        N = 3  # the number of layer in each building block
        self.m_input = ConvInput(1, C, N, filterSize=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))     # inputSize = output

        self.m_downList, outputSize, Cnew = self.addDownBBListWithMoreFilters(self.m_inputSize, C, 3, N)  # outputSize ={2*16*16}

        self.m_downList.append( DownBB(Cnew, 2*Cnew, filter1st=(2,3,3), stride=(2,2,2), nLayers=N))      # outpusSize =(1*7*7}
        outputSize = self.getConvOutputTensorSize(outputSize,(2,3,3), (2,2,2), (0,0,0))
        Cnew = 2*Cnew
        self.m_bottleNeckSize = (Cnew,) + outputSize

        # for response prediction
        nFilters  = 20;
        self.m_11Conv = BN_ReLU_Conv(Cnew, nFilters, (1,1,1), (1,1,1), (0,0,0), False)           # outpusSize =(1*7*7}
        lenBn = nFilters*49
        self.m_fc11   = nn.Sequential(
                       nn.Linear(lenBn , lenBn),
                       nn.Dropout(p = self.m_dropoutProb),
                       nn.ReLU(inplace=True),
                       nn.Linear(lenBn, lenBn),
                       nn.Dropout(p = self.m_dropoutProb),
                       nn.ReLU(inplace=True),
                       nn.Linear(lenBn, Kr))

        # for segmentation reconstruction
        outputSize = self.getConvTransposeOutputTensorSize(outputSize, (2,3,3), (2,2,2), (0,0,0))
        self.m_upList, outputSize, Cnewnew = self.addUpBBListWithLessFilters(outputSize, Cnew//2, 3, N)  # outputSize = 23*127*127
        self.m_upList.insert(0, UpBB(Cnew, Cnew//2, filter1st=(2,3,3), stride=(2,2,2), nLayers=N))
        Cnew = Cnewnew

        self.m_upOutput = nn.Conv3d(Cnew, Kup, (1,1,1), stride=(1,1,1))                   # outputSize = 23*127*127

