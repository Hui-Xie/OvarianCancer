
import torch.nn as nn
import torch

import sys
sys.path.append("../..")
from framework.BasicModel import BasicModel

sys.path.append(".")
from MobileNetV3_O import MobielNetV3_O
from VGG_O import VGG_O


class ResponseNet(BasicModel):
    def __init__(self, hps=None):
        '''
        inputSize: BxCxHxW
        '''
        super().__init__()
        self.hps = hps
        self.m_residualClassWeight = torch.tensor([1.0/item for item in hps.residudalClassPercent]).to(hps.device)
        self.m_chemoPosWeight = torch.tensor(hps.chemoClassPercent[0]/ hps.chemoClassPercent[1]).to(hps.device)
        self.m_optimalPosWeight = torch.tensor(hps.optimalClassPercent[0] / hps.optimalClassPercent[1]).to(hps.device)
        self.m_optimalClassWeight = torch.tensor([1.0/item for item in hps.optimalClassPercent]).to(hps.device)
        
        # before the mobileNet, get some higher level feature for each slice, by channel-wise conv.
        inC  = hps.inputChannels
        self.m_sliceConv = nn.Sequential(
            nn.Conv2d(inC, inC, kernel_size=3, stride=1, padding=1, groups=inC, bias=True),
            nn.BatchNorm2d(inC),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inC, inC, kernel_size=3, stride=1, padding=1, groups=inC, bias=True),
            nn.BatchNorm2d(inC),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inC, inC, kernel_size=3, stride=1, padding=1, groups=inC, bias=True),
            nn.BatchNorm2d(inC),
            nn.ReLU6(inplace=True)
        )

        self.m_featureNet =  eval(hps.featureNet)(hps.inputChannels, hps.outputChannels, hps=hps)

        if hps.predictHeads[0]:
            self.m_residualSizeHead = nn.Sequential(
                nn.Conv2d(hps.outputChannels, hps.widthResidualHead, kernel_size=1, stride=1, padding=0, bias=False)
                )

        if hps.predictHeads[1]:
            self.m_chemoResponseHead = nn.Sequential(
                nn.Conv2d(hps.outputChannels, hps.widthChemoHead, kernel_size=1, stride=1, padding=0, bias=False)
                )

        if hps.predictHeads[2]:
            self.m_ageHead = nn.Sequential(
                nn.Conv2d(hps.outputChannels, hps.widthAgeHead, kernel_size=1, stride=1, padding=0, bias=False)
                )

        if hps.predictHeads[3]:
            self.m_survivalHead = nn.Sequential(
                nn.Conv2d(hps.outputChannels, hps.widthSurvivalHead, kernel_size=1, stride=1, padding=0, bias=False)
                )

        if hps.predictHeads[4]:
            self.m_optimalResultHead = nn.Sequential(
                nn.Conv2d(hps.outputChannels, hps.widthOptimalResultHead, kernel_size=1, stride=1, padding=0, bias=False)
            )


    def forward(self, inputs, GTs=None):
        # compute outputs
        epsilon = 1e-8
        device = inputs.device
        B,_,_,_ = inputs.shape
        x = inputs

        x = self.m_sliceConv(x)
        x = self.m_featureNet(x)

        # initial values:
        residualLoss = torch.tensor(0.0,device=device)
        chemoLoss = torch.tensor(0.0, device=device)
        ageLoss = torch.tensor(0.0, device=device)
        survivalLoss = torch.tensor(0.0, device=device)
        optimalLoss = torch.tensor(0.0, device=device)

        residualPredict = torch.zeros(B, device=device)
        chemoPredict = torch.zeros(B, device=device)
        agePredict = torch.zeros(B, device=device)
        survivalPredict = torch.zeros(B, device=device)
        optimalPredict = torch.zeros(B, device=device)

        predictProb = torch.zeros(B, device=device)

        #residual tumor size
        if self.hps.predictHeads[0]:
            residualFeature = self.m_residualSizeHead(x)  # size: 1x4x1x1
            residualFeature = residualFeature.view(B, self.hps.widthResidualHead)
            residualPredict = torch.argmax(residualFeature,dim=1)-1 # from [0,1,2,3] to [-1,0,1,2]
            residualGT = (GTs['ResidualTumor'] + 1).to(device=device, dtype=torch.float32)  # from [-1,0,1,2] to [0,1,2,3]
            residualCELossFunc = nn.CrossEntropyLoss(weight=self.m_residualClassWeight)
            residualLoss = residualCELossFunc(residualFeature, residualGT)

        #chemo response:
        if self.hps.predictHeads[1]:
            chemoFeature = self.m_chemoResponseHead(x)
            chemoFeature = chemoFeature.view(B)
            predictProb = torch.sigmoid(chemoFeature)
            chemoPredict = (predictProb >= 0.50).int().view(B)  # a vector of [0,1]
            existLabel = torch.nonzero( (GTs['ChemoResponse'] != -100).int(),as_tuple=True)
            if (len(existLabel) >0) and (existLabel[0].nelement() >0): # -100 ignore index
                chemoFeature = chemoFeature[existLabel]
                chemoGT = GTs['ChemoResponse'][existLabel].to(device=device, dtype=torch.float32)
                chemoBCEFunc = nn.BCEWithLogitsLoss(pos_weight=self.m_chemoPosWeight)
                chemoLoss = chemoBCEFunc(chemoFeature, chemoGT)

        # age prediction:
        if self.hps.predictHeads[2]:
            ageFeature = self.m_ageHead(x)
            ageFeature = ageFeature.view(B, self.hps.widthAgeHead)
            ageRange = torch.tensor(list(range(0, self.hps.widthAgeHead)), dtype=torch.float32, device=device).view(1, self.hps.widthAgeHead)
            softmaxAge = nn.functional.softmax(ageFeature,dim=1)
            ageRange = ageRange.expand_as(softmaxAge)
            agePredict = (softmaxAge*ageRange).sum(dim=1)
            existLabel = torch.nonzero((GTs['Age'] != -100).int(), as_tuple=True)
            if (len(existLabel) >0) and (existLabel[0].nelement() >0):  # -100 ignore index
                existAgePrediction = agePredict[existLabel]
                ageGT = GTs['Age'][existLabel].to(device=device, dtype=torch.float32)  # range [0,100)
                # ageCELossFunc = nn.CrossEntropyLoss()
                # ageLoss += ageCELossFunc(ageFeature, ageGT) # CE
                ageLoss += torch.pow(existAgePrediction-ageGT, 2).mean()  # MSE

        # survival time:
        if self.hps.predictHeads[3]:
            survivalFeature = self.m_survivalHead(x)
            survivalFeature = survivalFeature.view(B, self.hps.widthSurvivalHead)
            survivalPredict = torch.sigmoid(survivalFeature).sum(dim=1)  # this is a wonderful idea!!!

            existLabel = torch.nonzero((GTs['SurvivalMonths'] != -100).int(), as_tuple=True)
            if (len(existLabel) >0) and (existLabel[0].nelement() >0):  # -100 ignore index
                # all uses censor conditions
                z = GTs['SurvivalMonths'][existLabel]
                sFeature = survivalFeature[existLabel,:]
                sPredict = survivalPredict[existLabel]
                survivalLoss += torch.pow(sPredict - z, 2).mean()  # MSE loss

                z = z.int()
                # survival curve fitting loss
                # if 1 == GTs['Censor']:
                curveLoss = torch.tensor(0.0, device=device)
                for i in range(len(z)):
                    sfLive = sFeature[i,0:z].clone() # sf: survival feature
                    curveLoss -= nn.functional.logsigmoid(sfLive).sum()  # use logsigmoid to avoid nan

                    # normalize expz_i and P[i]
                    R = self.hps.widthSurvivalHead- z[i] # expRange

                    z_i = -torch.tensor(list(range(0,R)), dtype=torch.float32, device=device).view(R)
                    Sz_i = nn.functional.softmax(z_i, dim=0) # make sure sum=1
                    # While log_softmax is mathematically equivalent to log(softmax(x)),
                    # doing these two operations separately is slower, and numerically unstable.
                    logSz_i = nn.functional.log_softmax(z_i,dim=0)  # logSoftmax

                    sfCurve = sFeature[i, z[i]:self.hps.widthSurvivalHead].clone()

                    curveLoss += (Sz_i*(logSz_i - nn.functional.logsigmoid(sfCurve)
                                           + torch.log(torch.sigmoid(sfCurve).sum()+epsilon) )).sum()
                curveLoss /=len(z)
                survivalLoss += curveLoss

                # for 0 == GTs['Censor']
                #    survivalLoss += survivalFeature[z+1:].sum()- nn.functional.logsigmoid(survivalFeature).sum()

        if self.hps.predictHeads[4]:
            optimalFeature = self.m_optimalResultHead(x)

            # for outputHeadWidth =1 and use sigmoid
            optimalFeature = optimalFeature.view(B)
            predictProb = torch.sigmoid(optimalFeature)
            optimalPredict = (predictProb >= 0.50).int().view(B)  # a vector of [0,1]
            optimalGT = GTs['OptimalResult'].to(device=device, dtype=torch.float32)
            optimalBCEFunc = nn.BCEWithLogitsLoss(pos_weight=self.m_optimalPosWeight)
            optimalLoss = optimalBCEFunc(optimalFeature, optimalGT)

            
            #  for outputHeadWidth =2 and use softmax+ BCELoss
            '''
            optimalFeature = optimalFeature.view(B, self.hps.widthOptimalResultHead)
            optimalSoftmax = nn.functional.softmax(optimalFeature, dim=1)
            predictProb = optimalSoftmax[:, 1]
            optimalPredict = torch.argmax(optimalFeature,dim=1)
            optimalGT = GTs['OptimalResult'].to(device=device, dtype=torch.float32)
            optimalBatchWeight = torch.zeros_like(optimalGT)
            for i in range(B):
                optimalBatchWeight[i] = self.m_optimalClassWeight[int(optimalGT[i])]
            optimalBCEFunc = nn.BCELoss(weight=optimalBatchWeight)
            optimalLoss = optimalBCEFunc(predictProb, optimalGT)
            '''


        loss = residualLoss + chemoLoss + ageLoss+ survivalLoss + optimalLoss
        if torch.isnan(loss) or torch.isinf(loss):  # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return residualPredict, residualLoss, chemoPredict, chemoLoss, agePredict, ageLoss, survivalPredict, survivalLoss, optimalPredict, optimalLoss, predictProb
