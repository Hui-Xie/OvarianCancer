from BasicModel import BasicModel
from ResNeXtBlock import ResNeXtBlock
import torch.nn as nn
import torch

# ResNeXt based Attention Net

class ResAttentionNet(BasicModel):
    def __init__(self):
        super().__init__()
        # For input image size: 140*251*251 (zyx)
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(140, 128, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(128, 128, nGroups=32, withMaxPooling=False),
                        ResNeXtBlock(128, 256, nGroups=32, withMaxPooling=False)
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
        self.m_fc1    = nn.Linear(2048, 2)


    def forward(self, x):
        x = self.m_stage1(x)
        x = self.m_stage2(x)
        x = self.m_stage3(x)
        x = self.m_stage4(x)
        x = self.m_stage5(x)
        x = self.m_stage6(x)
        x = self.m_avgPool(x)
        x = torch.reshape(x, (x.shape[0], x.numel() // x.shape[0]))
        x = self.m_fc1(x)
        return x
