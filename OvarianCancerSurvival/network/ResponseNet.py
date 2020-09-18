
import torch.nn as nn
import torch
import math

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
        self.m_residualClassWeight = torch.tensor([1.0/item for item in hps.residudalClassPercent]).to(hps.device)
        self.m_chemoClassWeight = torch.tensor([1.0/item for item in hps.chemoClassPercent]).to(hps.device)

        self.m_mobilenet = MobileNetV3(hps.inputChannels)
        self.m_residualSizeHead = nn.Conv2d(1280, hps.widthResidualHead, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_chemoResponseHead = nn.Conv2d(1280, hps.widthChemoHead, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_ageHead = nn.Conv2d(1280, hps.widthAgeHead, kernel_size=1, stride=1, padding=0, bias=False)
        self.m_survivalHead = nn.Conv2d(1280, hps.widthSurvialHead, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, inputs, GTs=None):
        # compute outputs
        e = 1e-8
        device = inputs.device
        x = inputs

        x = self.m_mobilenet(x)
        residualFeature = self.m_residualSizeHead(x) # size: 1x4x1x1
        chemoFeature = self.m_chemoResponseHead(x)
        ageFeature = self.m_ageHead(x)
        survivalFeature = self.m_survivalHead(x)

        #residual tumor size
        residualFeature = residualFeature.view(1, self.hps.widthResidualHead)
        residualPredict = torch.argmax(residualFeature)-1 # from [0,1,2,3] to [-1,0,1,2]
        residualGT = torch.tensor(GTs['ResidualTumor'] + 1).to(device)  # from [-1,0,1,2] to [0,1,2,3]
        residualCELossFunc = nn.CrossEntropyLoss(weight=self.m_residualClassWeight)
        residualLoss = residualCELossFunc(residualFeature, residualGT)

        #chemo response:
        chemoFeature = chemoFeature.view(1, self.hps.widthChemoHead)
        chemoPredict = torch.argmax(chemoFeature) # [0,1]
        chemoLoss = 0.0
        if GTs['ChemoResponse'] != -100: # -100 ignore index
            chemoGT = torch.tensor(GTs['ChemoResponse']).to(device)  # [0,1]
            chemoCELossFunc = nn.CrossEntropyLoss(weight=self.m_chemoClassWeight)
            chemoLoss = chemoCELossFunc(chemoFeature, chemoGT)

        # age prediction:
        ageFeature = ageFeature.view(1, self.hps.widthAgeHead)
        agePredict = torch.argmax(ageFeature)  # range [0,100)
        ageLoss = 0.0
        if GTs['Age'] != -100:  # -100 ignore index
            ageGT = torch.tensor(GTs['Age']).to(device)  # range [0,100)
            ageCELossFunc = nn.CrossEntropyLoss()
            ageLoss = ageCELossFunc(ageFeature, ageGT)


        # survival time:
        survivalFeature = survivalFeature.view(self.hps.widthSurvialHead)
        P = torch.sigmoid(survivalFeature)  # survival prob
        survivalPredict = torch.argmax((P>= 0.5).int())  # argmax return the  position of the last maximum.
        survivalLoss = 0.0
        if (GTs['SurvivalMonths'] != -100) and (GTs['Censor'] != -100):
            z = int(GTs['SurvivalMonths']+0.5)
            if 1 == GTs['Censor']:
                PLive = P[0:z].clone()
                survivalLoss -= torch.log(PLive).sum()

                # normalize expz_i and P[i]
                R = self.hps.widthSurvialHead- z # expRange
                expz_i = -torch.tensor(list(range(0,R)), dtype=torch.float32, device=device)
                expz_i = torch.exp(expz_i)
                expSum = expz_i.sum()
                expz_i = expz_i/expSum

                PCurve = P[z:self.hps.widthSurvialHead].clone()
                PCurveSum = PCurve.sum()
                PCurve = PCurve/PCurveSum

                survivalLoss += (expz_i*torch.log(PCurve/expz_i)).sum()

            else: # 0 == GTs['Censor']
                PLive = P[0:z+1].clone()
                survivalLoss -= torch.log(PLive).sum()

                PCurve = P[z+1:self.hps.widthSurvialHead].clone()
                survivalLoss -= torch.log(1.0-PCurve).sum()


        return residualPredict, residualLoss, chemoPredict, chemoLoss, agePredict, ageLoss, survivalPredict, survivalLoss
