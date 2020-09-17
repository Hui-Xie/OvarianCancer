
import torch.nn as nn
import torch

import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel
from framework.MobileNetV3 import MobileNetV3

class ResponseNet(BasicModel):
    def __init__(self, hps=None):
        '''
        inputSize: BxCxHxW
        '''
        super().__init__()
        self.hps = hps
        self.m_resdualClassWeight = torch.tensor([1.0/item for item in hps.residudalClassPercent]).to(hps.device)
        self.m_chemoClassWeight = torch.tensor([1.0/item for item in hps.chemoClassPercent]).to(hps.device)

        self.m_mobilenet = MobileNetV3(hps.inputChannels)
        self.m_residualSizeHead = nn.Conv2d(1280, hps.widthResidualHead, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_survivalHead = nn.Conv2d(1280, hps.widthSurvialHead, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_chemoResponseHead = nn.Conv2d(1280, hps.widthChemoHead, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_ageHead = nn.Conv2d(1280, hps.widthAgeHead, kernel_size=1, stride=1, padding=0, bias=False)



    def forward(self, inputs, GTs=None):
        # compute outputs
        e = 1e-8
        device = inputs.device
        x = inputs

        x = self.m_mobilenet(x)
        residualFeature = self.m_residualSizeHead(x) # size: 1x4x1x1
        survivalFeature = self.m_survivalHead(x)
        chemoResponseFeature = self.m_chemoResponseHead(x)
        ageFeature = self.m_ageHead(x)

        #todo: implement Head and its loss computation
        #residual tumor size
        residualFeature = residualFeature.view(1, self.hps.widthResidualHead)
        residualGT = torch.tensor(GTs['ResidualTumor']+1).to(device) # from [-1,0,1,2] to [0,1,2,3]
        residualPredict = torch.argmax(residualFeature)-1 # from [0,1,2,3] to [-1,0,1,2]
        residualCELossFunc = nn.CrossEntropyLoss(weight=self.m_resdualClassWeight)
        residualLoss = residualCELossFunc(residualFeature, residualGT)









        return residualPredict, residualLoss
