
'''
input: BxSlicePerEye, inputC, H, W
output: B, outputC, 1,1  to classification head

'''
import torch
import torch.nn as nn
import sys
sys.path.append("../..")

from framework.BasicModel import  BasicModel
from framework.Conv2DFeatureNet import  Conv2DFeatureNet

sys.path.append(".")
from MobileNetV3_OCT2SysD import MobileNetV3_OCT2SysD
import math

class OCT2SysD_Net(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps

        if hps.featureNet == "MobileNetV3_OCT2SysD":
            self.m_featureNet = eval(hps.featureNet)(hps=hps)
        elif hps.featureNet == "Conv2DFeatureNet":
            self.m_featureNet = eval(hps.featureNet)(hps.inputChannels, hps.nStartFilters, hps.nLayers, hps.outputChannels, hps.inputActivation)
        else:
            print("featureNet parameter error")
            assert False


        # after m_conv2d_1, tensor need global mean on H and W dimension.

        if 2 == len(hps.classifierWidth):
            self.m_classifier = nn.Sequential(
                nn.Conv2d(hps.outputChannels, hps.classifierWidth[0], kernel_size=1, stride=1, padding=0, bias=True),
                nn.Hardswish(),
                nn.Dropout2d(p=hps.dropoutRate, inplace=False), # here it must use inplace =False
                nn.Conv2d(hps.classifierWidth[0], hps.classifierWidth[1], kernel_size=1, stride=1, padding=0, bias=False)
                # if 1 == hps.classifierWidth[2], final linear layer does not need bias;
                )
        else: # 1 == len(hps.classifierWidth)
            assert 1 == len(hps.classifierWidth)
            self.m_classifier = nn.Sequential(
                nn.Hardswish(),
                nn.Dropout2d(p=hps.dropoutRate, inplace=False),  # here it must use inplace =False
                nn.Conv2d(hps.outputChannels, hps.classifierWidth[0], kernel_size=1, stride=1, padding=0, bias=False)
                # if 1 == hps.classifierWidth[0], final linear layer does not need bias;
            )

    def forward(self, x):
        '''

        :param x:  B*S, 3,H,W
        :return:   B,C, 1,1
        '''
        x = self.m_featureNet(x)

        # mean at H and W dimension
        B, C, H, W = x.shape
        if self.hps.globalPooling == "average":
            x = x.mean(dim=(-1, -2), keepdim=True) # B,C,1,1
        elif self.hps.globalPooling == "max":
            x = nn.AdaptiveMaxPool2d((1,1))(x)  # B,C, 1,1
        elif self.hps.globalPooling == "IQR_std":
            # Non-traditionally: Use IQR+std on H,and W dimension, each feature plane
            # IQR+std measures the spread of feature values in each channel
            xStd = torch.std(x, dim=(2, 3), keepdim=True)  # size: B,hps.outputChannels, 1,1
            xFeatureFlat = x.view(B, self.hps.outputChannels, -1)
            xSorted, _ = torch.sort(xFeatureFlat, dim=-1)
            B, C, N = xSorted.shape
            Q1 = N // 4
            Q3 = N * 3 // 4
            # InterQuartile Range
            IQR = (xSorted[:, :, Q3] - xSorted[:, :, Q1]).unsqueeze(dim=-1).unsqueeze(dim=-1)
            x = xStd + IQR
        else:
            print(f"Current do not support pooling: {self.hps.globalPooling}")
            assert  False

        x = self.m_classifier(x)
        return x

    def computeBinaryLoss(self, x, GTs=None, GTKey="", posWeight=None):
        '''
         For binary logits loss
        :param x: B,C,1,1
        :param GTs:
        :return:
        '''
        B,C,H,W = x.shape
        device = x.device

        x = x.view(B)
        predictProb = torch.sigmoid(x)
        if self.hps.TTA and ((self.m_status == "validation") or (self.m_status == "test")): # TTA
            predictProb = torch.mean(predictProb, dim=0, keepdim=True)
            B = 1
        predict = (predictProb >= 0.50).int().view(B)  # a vector of [0,1]
        GT = GTs[GTKey].to(device=device, dtype=torch.float32)
        bceFunc = nn.BCEWithLogitsLoss(pos_weight=posWeight.to(device))
        loss = bceFunc(x, GT.view_as(x))

        if torch.isnan(loss) or torch.isinf(loss):  # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return predict, predictProb, loss






