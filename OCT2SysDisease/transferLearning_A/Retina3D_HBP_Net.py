


import torch
import torch.nn as nn
import torchvision.models as models
import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel

class Retina3D_HBP_Net(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        self.posWeight = torch.tensor(hps.class01Percent[0] / hps.class01Percent[1]).to(hps.device)

        # load pretrained network
        self.ResNeXt = models.resnext101_32x8d(pretrained=True)
        # modify input layer and parameter
        C = hps.inputChannels
        conv1WeightData =  self.ResNeXt.conv1.weight.data.clone() # in size: 64x3x7x7
        self.ResNeXt.conv1 = nn.Conv2d(C, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # self.conv1.weith.data in size: 64xCx7x7
        with torch.no_grad():
            for c in range(C):
                self.ResNeXt.conv1.weight.data[:,c,:,:] = conv1WeightData[:,c%3,:,:]

        # modify the output layer and parameter
        N = hps.outputChannels
        fcWeightData = self.ResNeXt.fc.weight.data.clone() # in size: 1000x2048
        fcBiasData = self.ResNeXt.fc.bias.data.clone()  # in size: 1000
        self.ResNeXt.fc = nn.Linear(in_features=2048, out_features=N, bias=True)
        with torch.no_grad():
            for n in range(N):
                self.ResNeXt.fc.weight.data[n,:] = fcWeightData[n%1000,:]
                self.ResNeXt.fc.bias.data[n] = fcBiasData[n%1000]

        print(f"finish load pretrained network from Pytorch website, and parameter modification")

    def forward(self,x,t):
        x = self.ResNeXt(x)
        x = x.squeeze(dim=-1)  # B
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.posWeight)
        loss = criterion(x, t)
        return x, loss

