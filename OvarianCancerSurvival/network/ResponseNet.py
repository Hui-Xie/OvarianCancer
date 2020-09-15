
import torch.nn as nn

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
        residualFeature = self.m_residualSizeHead(x)
        survivalFeature = self.m_survivalHead(x)
        chemoResponseFeature = self.m_chemoResponseHead(x)
        ageFeature = self.m_ageHead(x)

        #todo: implement Head and its loss computation.

        return None
