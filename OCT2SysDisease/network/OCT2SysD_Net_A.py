
'''
input: BxSlicePerEye, inputC, H, W
output: B, outputC, 1,1  to classification head

'''
import torch
import torch.nn as nn
import sys
sys.path.append("../..")
from framework.SE_BottleNeck import  V3Bottleneck
from framework.BasicModel import  BasicModel
import math

class OCT2SysD_Net_A(BasicModel):
    def __init__(self, hps=None):
        super().__init__()
        self.hps = hps
        inC = 16  # input Channel number for bottleneck
        self.m_inputConv = nn.Sequential(
            nn.Conv2d(hps.inputChannels, inC, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(inC),
            nn.ReLU6(inplace=True)
        )

        bottleneckConfig = [
            # kernel, expandSize, outputChannel,  SE,   NL,  stride,
            # modify all activations to ReLU6
            [3, 16, 16, False, 'RE', 1],
            [3, 64, 24, False, 'RE', 2],
            [3, 72, 24, False, 'RE', 1],
            [5, 72, 40, True, 'RE', 2],
            [5, 120, 40, True, 'RE', 1],
            [5, 120, 40, True, 'RE', 1],
            [3, 240, 80, False, 'RE', 2],
            [3, 200, 80, False, 'RE', 1],
            [3, 184, 80, False, 'RE', 1],
            [3, 184, 80, False, 'RE', 1],
            [3, 480, 112, True, 'RE', 1],
            [3, 672, 112, True, 'RE', 1],
            [5, 672, 160, True, 'RE', 2],
            [5, 960, 160, True, 'HS', 1],
            [5, 960, 160, True, 'HS', 1],
        ]  # for MobileNet V3 big model
        self.m_bottleneckList = nn.ModuleList()
        for kernel, expandSize, outC, SE, NL, stride in bottleneckConfig:
            self.m_bottleneckList.append(
                V3Bottleneck(inC, outC, kernel=kernel, stride=stride, expandSize=expandSize, SE=SE, NL=NL))
            inC = outC

        self.hps.spaceD = hps.slicesPerEye
        if hps.bothEyes:
            self.hps.spaceD = hps.slicesPerEye *2

        self.m_featureConv = nn.Sequential(
            nn.Conv2d(inC, hps.outputChannels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hps.outputChannels),
            nn.ReLU6(inplace=True)
        )

        # after m_conv2d_1, tensor need global mean on Slice and H and W dimension.

        self.m_classifier = nn.Sequential(
            nn.Conv2d(hps.outputChannels, hps.classifierWidth[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=hps.dropoutRate, inplace=False),
            nn.Conv2d(hps.classifierWidth[0], hps.classifierWidth[1], kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=hps.dropoutRate, inplace=False),
            nn.Conv2d(hps.classifierWidth[1], hps.classifierWidth[2], kernel_size=1, stride=1, padding=0, bias=False)
            )

        self._initializeWeights()

    def forward(self, x):
        '''

        :param x:  B*S, 3,H,W
        :return:   B,C, 1,1
        '''
        x = self.m_inputConv(x)
        for bottle in self.m_bottleneckList:
            x = bottle(x)
        x = self.m_featureConv(x)

        # mean at slice, H and W dimension
        B, C, H, W = x.shape
        x = x.view(B // self.hps.spaceD, self.hps.spaceD, C, H, W)
        x = x.mean(dim=(1,3, 4), keepdim=True).squeeze(dim=1) # B,C,1,1

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

    def _initializeWeights(self):
        '''
        refer to https://github.com/xiaomi-automl/MoGA/blob/master/models/MoGA_A.py

        :return:
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(0)  # fan-out
                init_range = 1.0 / math.sqrt(n)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()




