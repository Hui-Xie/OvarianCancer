from BasicModel import BasicModel
from ResNeXtBlock import ResNeXtBlock
import torch.nn as nn
import torch

# ResNeXt based Attention Net

class ResAttentionNet(BasicModel):
    def __init__(self):
        super().__init__()
        # For input image size: 140*251*251 (zyx)
        # at July 27 16:20, 2019, reduce network parameters again from  1.23 million parameters to
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 128, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 160, nGroups=32, poolingLayer=None)
                        )  # ouput size: 160*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None)
                        ) # ouput size: 160*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None)
                        ) # output size: 160*128*128
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None)
                        )  # output size: 160*64*64
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None)
                        )  # output size: 160*32*32
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None)
                        )  # output size: 160*16*16
        self.m_stage6 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(160, 160, nGroups=32, poolingLayer=None)
                        )  # output size: 160*8*8
        self.m_avgPool= nn.AvgPool2d(8)
        self.m_fc1    = nn.Linear(160, 1, bias=False)  # for sigmoid output, one number

        """
        # For input image size: 140*251*251 (zyx)
        # at July 27 14:46, 2019, reduce network parameters again with 1.23 million parameters: : log_ResAttention_CV0_20190727_1459.txt
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 128, nGroups=20, withMaxPooling=False),
                        ResNeXtBlock(128, 128, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(128, 160, nGroups=32, withMaxPooling=False)
                        )  # ouput size: 160*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False)
                        ) # ouput size: 160*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False)
                        ) # output size: 160*128*128
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False)
                        )  # output size: 160*64*64
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False)
                        )  # output size: 160*32*32
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False)
                        )  # output size: 160*16*16
        self.m_stage6 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False)
                        )  # output size: 160*8*8
        self.m_avgPool= nn.AvgPool2d(8)
        self.m_fc1    = nn.Linear(160, 1, bias=False)  # for sigmoid output, one number
        
        """


        """
        #  log_ResAttention_CV0_20190727_0840.txt with parameters of 3.14 millions
        # For input image size: 140*251*251 (zyx)
        # at July 27, 2019, reduce network parameters
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 128, nGroups=20, withMaxPooling=False),
                        ResNeXtBlock(128, 128, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(128, 160, nGroups=32, withMaxPooling=False)
                        )  # ouput size: 160*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(160, 160, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(160, 192, nGroups=32, withMaxPooling=False)
                        ) # ouput size: 192*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(192, 192, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(192, 192, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(192, 224, nGroups=32, withMaxPooling=False)
                        ) # output size: 224*128*128
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(224, 224, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(224, 224, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(224, 256, nGroups=32, withMaxPooling=False)
                        )  # output size: 256*64*64
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(256, 256, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(256, 288, nGroups=32, withMaxPooling=False)
                        )  # output size: 288*32*32
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(288, 288, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(288, 288, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(288, 320, nGroups=32, withMaxPooling=False)
                        )  # output size: 320*16*16
        self.m_stage6 = nn.Sequential(
                        ResNeXtBlock(320, 320, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(320, 320, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(320, 352, nGroups=32, withMaxPooling=False)
                        )  # output size: 352*8*8
        self.m_avgPool= nn.AvgPool2d(8)
        self.m_fc1    = nn.Linear(352, 1, bias=False)  # for sigmoid output, one number
        
        """


        """
        # this is network in 20190726 with 7 million parameters:
        # For input image size: 140*251*251 (zyx)
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 128, nGroups=20, withMaxPooling=False),
                        ResNeXtBlock(128, 128, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(128, 256, nGroups=32, withMaxPooling=False)
                        )  # ouput size: 256*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(256, 256, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(256, 256, nGroups=32, withMaxPooling=False)
                        ) # ouput size: 256*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(256, 256, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(256, 512, nGroups=32, withMaxPooling=False)
                        ) # output size: 512*128*128
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(512, 512, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(512, 1024, nGroups=32, withMaxPooling=False)
                        )  # output size: 1024*64*64
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(1024, 1024, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(1024, 2048, nGroups=32, withMaxPooling=False)
                        )  # output size: 2048*32*32
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(2048, 2048, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(2048, 2048, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(2048, 2048, nGroups=32, withMaxPooling=False)
                        )  # output size: 2048*16*16
        self.m_stage6 = nn.Sequential(
                        ResNeXtBlock(2048, 2048, nGroups=32, withMaxPooling=True),
                        ResNeXtBlock(2048, 2048, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(2048, 2048, nGroups=32, withMaxPooling=False)
                        )  # output size: 2048*8*8
        self.m_avgPool= nn.AvgPool2d(8)
        self.m_fc1    = nn.Linear(2048, 1, bias=False)  # for sigmoid output, one number
        
        """


    def forward(self, x):
        x = self.m_stage0(x)
        x = self.m_stage1(x)
        x = self.m_stage2(x)
        x = self.m_stage3(x)
        x = self.m_stage4(x)
        x = self.m_stage5(x)
        x = self.m_stage6(x)
        x = self.m_avgPool(x)
        x = torch.reshape(x, (x.shape[0], x.numel() // x.shape[0]))
        x = self.m_fc1(x)
        x = x.squeeze(dim=1)
        return x
