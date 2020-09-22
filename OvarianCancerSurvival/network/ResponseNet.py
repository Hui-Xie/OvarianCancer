
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
        self.m_chemoPosWeight = torch.tensor(hps.chemoClassPercent[0]/ hps.chemoClassPercent[1]).to(hps.device)

        self.m_mobilenet = MobileNetV3(hps.inputChannels, hps.outputChannelsMobileNet)
        self.m_layerNormAfterMean =  nn.Sequential(
            nn.LayerNorm([hps.outputChannelsMobileNet, 1, 1], elementwise_affine=False),
            nn.Hardswish()
            )

        if hps.predictHeads[0]:
            self.m_residualSizeHead = nn.Sequential(
                nn.Conv2d(hps.outputChannelsMobileNet, hps.widthResidualHead, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LayerNorm([hps.widthResidualHead, 1, 1], elementwise_affine=False)
                )

        if hps.predictHeads[1]:
            self.m_chemoResponseHead = nn.Sequential(
                nn.Conv2d(hps.outputChannelsMobileNet, hps.widthChemoHead, kernel_size=1, stride=1, padding=0, bias=False)
                )

        if hps.predictHeads[2]:
            self.m_ageHead = nn.Sequential(
                nn.Conv2d(hps.outputChannelsMobileNet, hps.widthAgeHead, kernel_size=1, stride=1, padding=0, bias=False),
                nn.LayerNorm([hps.widthAgeHead, 1, 1], elementwise_affine=False)
                )

        if hps.predictHeads[3]:
            self.m_survivalHead = nn.Sequential(
                nn.Conv2d(hps.outputChannelsMobileNet, hps.widthSurvivalHead, kernel_size=1, stride=1, padding=0, bias=False)
                )


    def forward(self, inputs, GTs=None):
        # compute outputs
        epsilon = 1e-8
        device = inputs.device
        x = inputs

        x = self.m_mobilenet(x)
        x = self.m_layerNormAfterMean(x)

        # initial values:
        residualLoss = torch.tensor(0.0,device=device)
        chemoLoss = torch.tensor(0.0, device=device)
        ageLoss = torch.tensor(0.0, device=device)
        survivalLoss = torch.tensor(0.0, device=device)

        residualPredict = torch.tensor(0.0, device=device)
        chemoPredict = torch.tensor(0.0, device=device)
        agePredict = torch.tensor(0.0, device=device)
        survivalPredict = torch.tensor(0.0, device=device)

        #residual tumor size
        if self.hps.predictHeads[0]:
            residualFeature = self.m_residualSizeHead(x)  # size: 1x4x1x1
            residualFeature = residualFeature.view(1, self.hps.widthResidualHead)
            residualPredict = torch.argmax(residualFeature)-1 # from [0,1,2,3] to [-1,0,1,2]
            residualGT = torch.tensor(GTs['ResidualTumor'] + 1).to(device)  # from [-1,0,1,2] to [0,1,2,3]
            residualCELossFunc = nn.CrossEntropyLoss(weight=self.m_residualClassWeight)
            residualLoss = residualCELossFunc(residualFeature, residualGT)

        #chemo response:
        if self.hps.predictHeads[1]:
            chemoFeature = self.m_chemoResponseHead(x)
            chemoFeature = chemoFeature.view(1,self.hps.widthChemoHead)
            chemoPredict = (chemoFeature > 0).int().view(self.hps.widthChemoHead) # [0,1]
            if GTs['ChemoResponse'] != -100: # -100 ignore index
                chemoGT = torch.tensor(GTs['ChemoResponse']).to(device=device, dtype=torch.float32).view(1, self.hps.widthChemoHead)  # [0,1]
                chemoBCEFunc = nn.BCEWithLogitsLoss(pos_weight=self.m_chemoPosWeight)
                chemoLoss = chemoBCEFunc(chemoFeature, chemoGT)

        # age prediction:
        if self.hps.predictHeads[2]:
            ageFeature = self.m_ageHead(x)
            ageFeature = ageFeature.view(1, self.hps.widthAgeHead)
            ageRange = torch.tensor(list(range(0, self.hps.widthAgeHead)), dtype=torch.float32, device=device).view(1, self.hps.widthAgeHead)
            softmaxAge = nn.functional.softmax(ageFeature,dim=1)
            agePredict = (softmaxAge*ageRange).sum().view(1)
            if GTs['Age'] != -100:  # -100 ignore index
                ageGT = torch.tensor(GTs['Age']).to(device)  # range [0,100)
                # ageCELossFunc = nn.CrossEntropyLoss()
                # ageLoss += ageCELossFunc(ageFeature, ageGT) # CE
                ageLoss += torch.pow(agePredict-ageGT, 2)  # MSE

        # survival time:
        if self.hps.predictHeads[3]:
            survivalFeature = self.m_survivalHead(x)
            survivalFeature = survivalFeature.view(self.hps.widthSurvivalHead)
            survivalPredict = torch.sigmoid(survivalFeature).sum()  # this is a wonderful idea!!!

            if (GTs['SurvivalMonths'] != -100) and (GTs['Censor'] != -100):
                z = int(GTs['SurvivalMonths']+0.5)
                survivalLoss += torch.pow(survivalPredict - z, 2)  # MSE loss

                # survival curve fitting loss
                if 1 == GTs['Censor']:
                    sfLive = survivalFeature[0:z].clone() # sf: survival feature
                    survivalLoss -= nn.functional.logsigmoid(sfLive).sum()  # use logsigmoid to avoid nan

                    # normalize expz_i and P[i]
                    R = self.hps.widthSurvivalHead- z # expRange
                    z_i = -torch.tensor(list(range(0,R)), dtype=torch.float32, device=device).view(R)
                    Sz_i = nn.functional.softmax(z_i, dim=0) # make sure sum=1
                    # While log_softmax is mathematically equivalent to log(softmax(x)),
                    # doing these two operations separately is slower, and numerically unstable.
                    logSz_i = nn.functional.log_softmax(z_i,dim=0)  # logSoftmax

                    sfCurve = survivalFeature[z:self.hps.widthSurvivalHead].clone()

                    survivalLoss += (Sz_i*(logSz_i - nn.functional.logsigmoid(sfCurve)
                                           + torch.log(torch.sigmoid(sfCurve).sum()+epsilon) )).sum()

                else: # 0 == GTs['Censor']
                    survivalLoss += survivalFeature[z+1:].sum()- nn.functional.logsigmoid(survivalFeature).sum()


        loss = residualLoss + chemoLoss + ageLoss+ survivalLoss
        if torch.isnan(loss):  # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return residualPredict, residualLoss, chemoPredict, chemoLoss, agePredict, ageLoss, survivalPredict, survivalLoss
