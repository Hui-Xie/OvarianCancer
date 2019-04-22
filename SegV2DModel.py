import torch
import torch.nn as nn
import torch.nn.functional as F

from SegVModel import SegVModel

#  2D model

class SegV2DModel(SegVModel):
    def __init__(self, F, K):
        """

        :param F: The number of filter of the first layer, and it is better 64.
        :param K: the final output classification number.

        """
        super().__init__()

        self.m_conv1 = nn.Conv2d(1, F, (5, 5), stride=(2, 2))  # inputSize: 281*281; output:F*139*139
        self.m_bn1 = nn.BatchNorm2d(F)
        self.m_conv2 = nn.Conv2d(F, 2*F, (3, 3), stride=(2, 2))  # output: 2F*69*69
        self.m_bn2 = nn.BatchNorm2d(2*F)
        self.m_conv3 = nn.Conv2d(2*F, 4*F, (5, 5), stride=(2, 2))  # output: 4F*33*33
        self.m_bn3 = nn.BatchNorm2d(4*F)
        self.m_conv4 = nn.Conv2d(4*F, 8*F, (5, 5), stride=(2, 2))  # output: 8F*15*15
        self.m_bn4 = nn.BatchNorm2d(8*F)
        self.m_conv5 = nn.Conv2d(8*F, 16*F, (3, 3), stride=(2, 2))  # output: 16F*7*7
        self.m_bn5 = nn.BatchNorm2d(16*F)
        self.m_conv6 = nn.Conv2d(16*F, 16*F, (3, 3), stride=(2, 2))  # output: 16F*3*3
        self.m_bn6 = nn.BatchNorm2d(16*F)

        self.m_convT6 = nn.ConvTranspose2d(16*F, 16*F, (3, 3), stride=(2, 2))  # output: 16F*7*7
        self.m_bnT6 = nn.BatchNorm2d(16*F)
        self.m_convT5 = nn.ConvTranspose2d(32*F, 8*F, (3, 3), stride=(2, 2))  # output: 8F*15*15
        self.m_bnT5 = nn.BatchNorm2d(8*F)
        self.m_convT4 = nn.ConvTranspose2d(16*F, 4*F, (5, 5), stride=(2, 2))  # output: 4F*33*33
        self.m_bnT4 = nn.BatchNorm2d(4*F)
        self.m_convT3 = nn.ConvTranspose2d(8*F, 2*F, (5, 5), stride=(2, 2))  # output: 2F*69*69
        self.m_bnT3 = nn.BatchNorm2d(2*F)
        self.m_convT2 = nn.ConvTranspose2d(4*F, F, (3, 3), stride=(2, 2))  # output:F*139*139
        self.m_bnT2 = nn.BatchNorm2d(F)
        self.m_convT1 = nn.ConvTranspose2d(2*F, F-1, (5, 5), stride=(2, 2))  # output:(F-1)*281*281
        self.m_bnT1 = nn.BatchNorm2d(F-1)
        self.m_conv0 = nn.Conv2d(F, K, (1, 1), stride=1)  # output:K*281*281

    def forward(self, x):
        # without residual link within layer

        x1 = self.m_dropout2d(F.relu(self.m_bn1(self.m_conv1(x))))  # Conv->BatchNorm->ReLU will keep half postive input.
        x2 = self.m_dropout2d(F.relu(self.m_bn2(self.m_conv2(x1))))
        x3 = self.m_dropout2d(F.relu(self.m_bn3(self.m_conv3(x2))))
        x4 = self.m_dropout2d(F.relu(self.m_bn4(self.m_conv4(x3))))
        x5 = self.m_dropout2d(F.relu(self.m_bn5(self.m_conv5(x4))))
        xc = self.m_dropout2d(F.relu(self.m_bn6(self.m_conv6(x5))))  # xc means x computing

        xc = self.m_dropout2d(F.relu(self.m_bnT6(self.m_convT6(xc))))
        xc = torch.cat((xc, x5), 1)                         # batchsize is in dim 0, so concatenate at dim 1.
        xc = self.m_dropout2d(F.relu(self.m_bnT5(self.m_convT5(xc))))
        xc = torch.cat((xc, x4), 1)
        xc = self.m_dropout2d(F.relu(self.m_bnT4(self.m_convT4(xc))))
        xc = torch.cat((xc, x3), 1)
        xc = self.m_dropout2d(F.relu(self.m_bnT3(self.m_convT3(xc))))
        xc = torch.cat((xc, x2), 1)
        xc = self.m_dropout2d(F.relu(self.m_bnT2(self.m_convT2(xc))))
        xc = torch.cat((xc, x1), 1)
        xc = self.m_dropout2d(F.relu(self.m_bnT1(self.m_convT1(xc))))
        xc = torch.cat((xc, x), 1)

        xc = self.m_conv0(xc)

        # return output
        return xc
