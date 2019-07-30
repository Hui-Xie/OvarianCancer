import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from BasicModel import BasicModel
from BuildingBlocks import *

#  2D model

class SegV2DModel(BasicModel):
    def __init__(self, C, K):
        """

        :param C: channels, The number of filters of the first layer, and it is better 64.
        :param K: the final output classification number.

        """
        super().__init__()
        if C <=1:
            print("Error: the number of filter in first layer is too small.")
            sys.exit(-1)

        N = 4                    # the number of layer in each building block in each lay of V model.
        # 3 is for denseNet, 4 is for residual links.
        Depth = 5                # the depth of V model

        self.m_input = ConvInput(1, C, N)                          # inputSize: 1*281*281; output:C*281*281

        self.m_down1 = DownBB(C, C, (5, 5), stride=(2, 2), nLayers=N)            # output:C*139*139
        self.m_down2 = DownBB(C, 2 * C, (3, 3), stride=(2, 2), nLayers=N)        # output: 2C*69*69
        self.m_down3 = DownBB(2 * C, 4 * C, (5, 5), stride=(2, 2), nLayers=N)    # output: 4C*33*33
        self.m_down4 = DownBB(4 * C, 8 * C, (5, 5), stride=(2, 2), nLayers=N)    # output: 8C*15*15
        self.m_down5 = DownBB(8 * C, 16 * C, (3, 3), stride=(2, 2), nLayers=N)   # output: 16C*7*7
        # self.m_down6 = Down2dBB(16 * C, 16 * C, (3, 3), convStride=(2, 2), nLayers=N)  # output: 16C*3*3

        # the bridges between encoder and decoder
        self.m_resPath1 = ResPath(C, C, Depth - 1)
        self.m_resPath2 = ResPath(2*C, 2*C, Depth - 2)
        self.m_resPath3 = ResPath(4*C, 4*C, Depth - 3)
        self.m_resPath4 = ResPath(8*C, 8*C, Depth - 4)

        # self.m_up6   = Up2dBB(16 * C, 16 * C, (3, 3), convStride=(2, 2), nLayers=N)    # output: 16C*7*7
        # self.m_up5   = Up2dBB(32 * C, 8 * C, (3, 3), convStride=(2, 2), nLayers=N)     # output: 8C*15*15
        self.m_up5   = UpBB(16 * C, 8 * C, (3, 3), stride=(2, 2), nLayers=N)  # output: 8C*15*15
        self.m_up4   = UpBB(16 * C, 4 * C, (5, 5), stride=(2, 2), nLayers=N)     # output: 4C*33*33
        self.m_up3   = UpBB(8 * C, 2 * C, (5, 5), stride=(2, 2), nLayers=N)      # output: 2C*69*69
        self.m_up2   = UpBB(4 * C, C, (3, 3), stride=(2, 2), nLayers=N)          # output:C*139*139
        self.m_up1   = UpBB(2 * C, C, (5, 5), stride=(2, 2), nLayers=N)          # output:C*281*281

        # self.m_outputBn = nn.BatchNorm2d(2 * C)
        # self.m_output = nn.Conv2d(2*C, K, (1, 1), convStride=1)             # output:K*281*281
        self.m_output = ConvOutput(2*C, 2*C, N, K)


        # ==== Old code for single conv in each layer of V model ==========
        # self.m_conv1 = nn.Conv2d(1, C, (5, 5), convStride=(2, 2))  # inputSize: 281*281; output:F*139*139
        # self.m_bn1 = nn.BatchNorm2d(C)
        # self.m_conv2 = nn.Conv2d(C, 2 * C, (3, 3), convStride=(2, 2))  # output: 2C*69*69
        # self.m_bn2 = nn.BatchNorm2d(2 * C)
        # self.m_conv3 = nn.Conv2d(2 * C, 4 * C, (5, 5), convStride=(2, 2))  # output: 4C*33*33
        # self.m_bn3 = nn.BatchNorm2d(4 * C)
        # self.m_conv4 = nn.Conv2d(4 * C, 8 * C, (5, 5), convStride=(2, 2))  # output: 8C*15*15
        # self.m_bn4 = nn.BatchNorm2d(8 * C)
        # self.m_conv5 = nn.Conv2d(8 * C, 16 * C, (3, 3), convStride=(2, 2))  # output: 16C*7*7
        # self.m_bn5 = nn.BatchNorm2d(16 * C)
        # self.m_conv6 = nn.Conv2d(16 * C, 16 * C, (3, 3), convStride=(2, 2))  # output: 16C*3*3
        # self.m_bn6 = nn.BatchNorm2d(16 * C)
        #
        # self.m_convT6 = nn.ConvTranspose2d(16 * C, 16 * C, (3, 3), convStride=(2, 2))  # output: 16C*7*7
        # self.m_bnT6 = nn.BatchNorm2d(16 * C)
        # self.m_convT5 = nn.ConvTranspose2d(32 * C, 8 * C, (3, 3), convStride=(2, 2))  # output: 8C*15*15
        # self.m_bnT5 = nn.BatchNorm2d(8 * C)
        # self.m_convT4 = nn.ConvTranspose2d(16 * C, 4 * C, (5, 5), convStride=(2, 2))  # output: 4C*33*33
        # self.m_bnT4 = nn.BatchNorm2d(4 * C)
        # self.m_convT3 = nn.ConvTranspose2d(8 * C, 2 * C, (5, 5), convStride=(2, 2))  # output: 2C*69*69
        # self.m_bnT3 = nn.BatchNorm2d(2 * C)
        # self.m_convT2 = nn.ConvTranspose2d(4 * C, C, (3, 3), convStride=(2, 2))  # output:C*139*139
        # self.m_bnT2 = nn.BatchNorm2d(C)
        # self.m_convT1 = nn.ConvTranspose2d(2 * C, C - 1, (5, 5), convStride=(2, 2))  # output:(C-1)*281*281
        # self.m_bnT1 = nn.BatchNorm2d(C - 1)
        # self.m_conv0 = nn.Conv2d(C, K, (1, 1), convStride=1)  # output:K*281*281

    def halfForward(self, input):
        x0 = self.m_input(input)
        x1 = self.m_down1(x0)
        x2 = self.m_down2(x1)
        x3 = self.m_down3(x2)
        x4 = self.m_down4(x3)
        x5 = self.m_down5(x4)
        return x5

    def forward(self, input):
        x0 = self.m_input(input)
        x1 = self.m_down1(x0)
        x2 = self.m_down2(x1)
        x3 = self.m_down3(x2)
        x4 = self.m_down4(x3)
        x5 = self.m_down5(x4)
        # x6 = self.m_dropout2d(self.m_down6(x5))

        # x = self.m_dropout2d(self.m_up6(x6))
        # x = self.m_dropout2d(self.m_up5(x, x5))
        x = self.m_up5(x5)
        x = self.m_up4(x, self.m_resPath4(x4))
        x = self.m_up3(x, self.m_resPath3(x3))
        x = self.m_up2(x, self.m_resPath2(x2))
        x = self.m_up1(x, self.m_resPath1(x1))

        #x = torch.cat((x,x0),1)
        #x = self.m_outputBn(x)
        x = self.m_output(x, x0)

        return x



