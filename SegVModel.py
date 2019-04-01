import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

batchSize = 4

class SegVModel (nn.Module):
    def __init__(self):
        super(SegVModel, self).__init__()
        global batchSize
        self.m_conv1 = nn.Conv3d(1,   32,  (5,5,5), stride=(2,2,2))   #inputSize: 21*281*281; output:32*9*139*139
        self.m_conv2 = nn.Conv3d(32,  64,  (5,3,3), stride=(2,2,2))   #output: 64*3*69*69
        self.m_conv3 = nn.Conc3d(64,  128, (3,5,5), stride=(2,2,2))   #output: 128*1*33*33
        self.m_conv4 = nn.Conv2d(128, 256, (5,5),   stride=(2,2))     #output: 256*15*15
        self.m_conv5 = nn.Conv2d(256, 512, (3,3),   stride=(2,2))     #output: 512*7*7
        self.m_conv6 = nn.Conv2d(512, 512, (3,3),   stride=(2, 2))    #output: 512*3*3

        self.m_convT6 = nn.ConvTranspose2d(512,  512, (3,3),  stride=(2, 2))  #output: 512*7*7
        self.m_convT5 = nn.ConvTranspose2d(1024, 256, (3,3),  stride=(2,2))   #output: 256*15*15
        self.m_convT4 = nn.ConvTranspose2d(512,  128, (5,5),  stride=(2,2))   #output: 128*33*33
        self.m_convT3 = nn.ConvTranspose3d(256,  64,  (3,5,5),stride=(2,2,2)) #output: 64*3*69*69
        self.m_convT2 = nn.ConvTranspose3d(128,  32,  (5,3,3),stride=(2,2,2)) #output:32*9*139*139
        self.m_convT1 = nn.ConvTranspose3d(64,   1,   (5,5,5),stride=(2,2,2)) #output:21*281*281
        self.m_convT0 = nn.Conv2d(42,  4,  (1,1), stride=1)                   #output:4*281*281

    def forward(self, *input):
        # input clip and normalization here
        # from 3D to 2D, there is squeeze
        pass


