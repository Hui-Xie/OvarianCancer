
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

class OCT2SysD_Net_A(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps

        if hps.featureNet == "MobileNetV3_OCT2SysD":
            self.m_featureNet = eval(hps.featureNet)(hps=hps)
        elif hps.featureNet == "Conv2DFeatureNet":
            self.m_featureNet = eval(hps.featureNet)(hps.inputChannels, hps.nStartFilters, hps.nLayers, hps.outputChannels)
        else:
            print("featureNet parameter error")
            assert False


        # after m_conv2d_1, tensor need global mean on H and W dimension.

        self.m_classifier = nn.Sequential(
            nn.Conv2d(hps.outputChannels, hps.classifierWidth[0], kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=hps.dropoutRate, inplace=False),
            nn.Conv2d(hps.classifierWidth[0], hps.classifierWidth[1], kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=hps.dropoutRate, inplace=False),
            nn.Conv2d(hps.classifierWidth[1], hps.classifierWidth[2], kernel_size=1, stride=1, padding=0, bias=False)
            # if 1 == hps.classifierWidth[2], final linear layer does not need bias;
            )

    def forward(self, x):
        '''

        :param x:  B*S, 3,H,W
        :return:   B,C, 1,1
        '''
        x = self.m_featureNet(x)

        # mean at H and W dimension
        B, C, H, W = x.shape
        x = x.mean(dim=(-1, -2), keepdim=True) # B,C,1,1

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
        predict = (predictProb >= 0.50).int().view(B)  # a vector of [0,1]
        GT = GTs[GTKey].to(device=device, dtype=torch.float32)
        bceFunc = nn.BCEWithLogitsLoss(pos_weight=posWeight.to(device))
        loss = bceFunc(x, GT)

        if torch.isnan(loss) or torch.isinf(loss):  # detect NaN
            print(f"Error: find NaN loss at epoch {self.m_epoch}")
            assert False

        return predict, predictProb, loss






