# this is a model built as May 1st 11:51, which got 78% test dice for primary

import torch
import torch.nn as nn
import sys

from SegVModel import SegVModel
from ModuleBuildingBlocks import *

#  2D model

class SegV2DModel_78(SegVModel):
    def __init__(self, C, K):
        """
        :param C: channels, The number of filters of the first layer, and it is better 64.
        :param K: the final output classification number.
        """
        super().__init__()
        if C <=1:
            print("Error: the number of filter in first layer is too small.")
            sys.exit(-1)

        N = 3 # conv layers
        # use skip2Residual

        self.m_input = ConvInput(1, C, N-1)                                        # inputSize: 1*281*281; output:C*281*281
        self.m_down1 = Down2dBB(C, C, (5, 5), stride=(2, 2), nLayers=N)            # output:C*139*139
        self.m_down2 = Down2dBB(C, 2 * C, (3, 3), stride=(2, 2), nLayers=N)        # output: 2C*69*69
        self.m_down3 = Down2dBB(2 * C, 4 * C, (5, 5), stride=(2, 2), nLayers=N)    # output: 4C*33*33
        self.m_down4 = Down2dBB(4 * C, 8 * C, (5, 5), stride=(2, 2), nLayers=N)    # output: 8C*15*15
        self.m_down5 = Down2dBB(8 * C, 16 * C, (3, 3), stride=(2, 2), nLayers=N)   # output: 16C*7*7
        self.m_down6 = Down2dBB(16 * C, 16 * C, (3, 3), stride=(2, 2), nLayers=N)  # output: 16C*3*3

        self.m_up6   = Up2dBB(16 * C, 16 * C, (3, 3), stride=(2, 2), nLayers=N)    # output: 16C*7*7
        self.m_up5   = Up2dBB(32 * C, 8 * C, (3, 3), stride=(2, 2), nLayers=N)     # output: 8C*15*15
        # self.m_up5   = Up2dBB(16 * C, 8 * C, (3, 3), stride=(2, 2), nLayers=N)     # output: 8C*15*15
        self.m_up4   = Up2dBB(16 * C, 4 * C, (5, 5), stride=(2, 2), nLayers=N)     # output: 4C*33*33
        self.m_up3   = Up2dBB(8 * C, 2 * C, (5, 5), stride=(2, 2), nLayers=N)      # output: 2C*69*69
        self.m_up2   = Up2dBB(4 * C, C, (3, 3), stride=(2, 2), nLayers=N)          # output:C*139*139
        self.m_up1   = Up2dBB(2 * C, C, (5, 5), stride=(2, 2), nLayers=N)          # output:C*281*281

        self.m_output = nn.Conv2d(2*C, K, (1, 1), stride=1)                        # output:K*281*281



    def forward(self, input):
        x0 = self.m_input(input)
        x1 = self.m_down1(x0)
        x2 = self.m_down2(x1)
        x3 = self.m_down3(x2)
        x4 = self.m_down4(x3)
        x5 = self.m_down5(x4)
        x6 = self.m_down6(x5)

        x = self.m_dropout2d(self.m_up6(x6))
        x = self.m_dropout2d(self.m_up5(x, x5))
        # x = self.m_dropout2d(self.m_up5(x5))
        x = self.m_dropout2d(self.m_up4(x, x4))
        x = self.m_dropout2d(self.m_up3(x, x3))
        x = self.m_dropout2d(self.m_up2(x, x2))
        x = self.m_dropout2d(self.m_up1(x, x1))

        x = torch.cat((x,x0),1)
        x = self.m_output(x)

        return x

