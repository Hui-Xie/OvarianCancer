from BasicModel import BasicModel
from ResNeXtBlock import ResNeXtBlock
from SpatialTransformer import SpatialTransformer
import torch.nn as nn
import torch
from draw2DArray import *

# ResNeXt based Attention Net

class ResAttentionNet(BasicModel):
    def forward(self, x):
        # filename = "/home/hxie1/Projects/OvarianCancer/trainLog/20190816_194148"
        # midSlice1 = x[0, 115,].clone()
        # display2DImage(midSlice1.cpu().detach().numpy(), "before STN", filename+"_BeforeSTN.png")
        # x = self.m_stn0(x)
        # midSlice2 = x[0, 115,].clone()
        # display2DImage(midSlice2.cpu().detach().numpy(), "after STN", filename+"_AfterSTN.png" )
        x = self.m_stage0(x)
        x = self.m_stage1(x)
        x = self.m_stage2(x)
        x = self.m_stage3(x)
        x = self.m_stage4(x)
        x = x+ self.m_stn4(x)
        x = self.m_stage5(x)
        x = x+ self.m_stn5(x)
        x = self.m_layersBeforeFc(x)
        x = torch.reshape(x, (x.shape[0], x.numel() // x.shape[0]))
        x = self.m_fc1(x)
        x = x.squeeze(dim=1)
        return x

    def __init__(self):
        super().__init__()
        # For input image size: 231*251*251 (zyx)
        # at Aug 16 12:09 , 2019, input of gaussian normalization with non-zero mean, with STN
        # add maxPool at each stage, and 1024 is the final conv filter number.
        #  add filter number in the model.
        # log:
        #
        # result:
        #
        self.m_useSpectralNorm = True
        self.m_useLeakyReLU = True
        # self.m_stn0    = SpatialTransformer(231, 64, 251, 251, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(231, 128, nGroups=33, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                        )  # ouput size: 128*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1), useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                        # SpatialTransformer(128, 32, 126, 126)
                        ) # ouput size: 256*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1), useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                        # SpatialTransformer(256, 64, 63, 63)
                        ) # output size: 512*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1), useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                        # SpatialTransformer(512, 64, 32, 32)
                        )  # output size: 1024*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1), useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(1024, 2048, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                        )  # output size: 2048*16*16
        self.m_stn4   = SpatialTransformer(2048, 512, 16, 16, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(2048, 2048, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1), useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(2048, 2048, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU),
                        ResNeXtBlock(2048, 4096, nGroups=32, poolingLayer=None, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
                        )  # output size: 4096*8*8
        self.m_stn5   = SpatialTransformer(4096, 512, 8, 8, useSpectralNorm=self.m_useSpectralNorm, useLeakyReLU=self.m_useLeakyReLU)
        self.m_layersBeforeFc=nn.Sequential(
                             nn.Conv2d(4096, 1024, kernel_size=8, stride=8, padding=0, bias=True),
                             nn.ReLU() if not self.m_useLeakyReLU else nn.LeakyReLU(),
                             nn.LocalResponseNorm(1024)   # normalization on 1024 channels.
                             ) # output size: 1024*1*1
        #if self.m_useSpectralNorm:
        #     self.m_layerBeforeFc = nn.utils.spectral_norm(self.m_layerBeforeFc)  # this costs a lot of memory.

        self.m_fc1    = nn.Linear(1024, 1, bias=True)  # for sigmoid output, one number
        #if self.m_useSpectralNorm:
        #     self.m_fc1 = nn.utils.spectral_norm(self.m_fc1)

        """
        super().__init__()
        # For input image size: 231*251*251 (zyx)
        # at Aug 13 10:47 , 2019, input of gaussian normalization with non-zero mean, without STN
        # add maxPool at each stage, and 1024 is the final conv filter number.
        #  add filter number in the model.
        # log:  log_ResAttention_CV0_20190813_111441.txt
        #
        # result: training loss is decreasing.
        #
        #self.m_stn    = SpatialTransformer(231,32, 251,251)
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(231, 128, nGroups=33, poolingLayer=None),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None)
                        )  # ouput size: 128*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(128, 32, 126, 126)
                        ) # ouput size: 256*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(256, 64, 63, 63)
                        ) # output size: 512*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(512, 64, 32, 32)
                        )  # output size: 1024*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(1024, 2048, nGroups=32, poolingLayer=None)
                        )  # output size: 2048*16*16
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(2048, 2048, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(2048, 2048, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(2048, 4096, nGroups=32, poolingLayer=None)
                        )  # output size: 4096*8*8
        self.m_layerBeforeFc=nn.Conv2d(4096, 1024, kernel_size=8, stride=8, padding=0, bias=False)

        self.m_fc1    = nn.Linear(1024, 1, bias=False)  # for sigmoid output, one number
        """

        """
        # For input image size: 231*251*251 (zyx)
        # at Aug 12 09am , 2019, input of gaussian normalization, without STN
        # add maxPool at each stage, and 1024 is the final conv filter number.
        # log: log_ResAttention_CV0_20190813_102504.txt
        #
        # result:
        #
        #self.m_stn    = SpatialTransformer(231,32, 251,251)
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(231, 32, nGroups=33, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 64, nGroups=8, poolingLayer=None)
                        )  # ouput size: 64*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 128, nGroups=16, poolingLayer=None)
                        # SpatialTransformer(128, 32, 126, 126)
                        ) # ouput size: 128*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(256, 64, 63, 63)
                        ) # output size: 256*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(512, 64, 32, 32)
                        )  # output size: 512*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*16*16
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*8*8
        self.m_layerBeforeFc=nn.Conv2d(1024, 1024, kernel_size=8, stride=8, padding=0, bias=False)

        self.m_fc1    = nn.Linear(1024, 1, bias=False)  # for sigmoid output, one number

        
        """


        """
        super().__init__()
        # For input image size: 231*251*251 (zyx)
        # at Aug 11 08:30, 2019, input of gaussian normalization, put STN before the network
        # add maxPool at each stage, and 1024 is the final conv filter number.
        # log:    log_ResAttention_CV2_20190811_083709.txt
        #         log_ResAttention_CV0_20190811_083630.txt
        # Result: program converge into majority prediction.
        #
        self.m_stage0 = nn.Sequential(
                        SpatialTransformer(231,32, 251,251),
                        ResNeXtBlock(231, 32, nGroups=33, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 64, nGroups=8, poolingLayer=None)

                        )  # ouput size: 64*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 128, nGroups=16, poolingLayer=None)
                        # SpatialTransformer(128, 32, 126, 126)
                        ) # ouput size: 128*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(256, 64, 63, 63)
                        ) # output size: 256*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(512, 64, 32, 32)
                        )  # output size: 512*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*16*16
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*8*8
        self.m_layerBeforeFc=nn.Conv2d(1024, 1024, kernel_size=8, stride=8, padding=0, bias=False)

        self.m_fc1    = nn.Linear(1024, 1, bias=False)  # for sigmoid output, one number
        
        
        """


        """
         # For input image size: 231*251*251 (zyx)
        # at Aug 11 02:20, 2019, A network without STN, but with input of gaussian normalization
        # add maxPool at each stage, and 1024 is the final conv filter number.
        # log:  log_ResAttention_CV2_20190811_022037.txt
        #       log_ResAttention_CV0_20190811_021952.txt
        #
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(231, 32, nGroups=33, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 64, nGroups=8, poolingLayer=None)
                        # SpatialTransformer(64,16, 251,251)
                        )  # ouput size: 64*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 128, nGroups=16, poolingLayer=None)
                        # SpatialTransformer(128, 32, 126, 126)
                        ) # ouput size: 128*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(256, 64, 63, 63)
                        ) # output size: 256*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None)
                        # SpatialTransformer(512, 64, 32, 32)
                        )  # output size: 512*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*16*16
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*8*8
        self.m_layerBeforeFc=nn.Conv2d(1024, 1024, kernel_size=8, stride=8, padding=0, bias=False)

        self.m_fc1    = nn.Linear(1024, 1, bias=False)  # for sigmoid output, one number
        
        """

        """
         # For input image size: 231*251*251 (zyx)
        # at Aug 10 11:56, 2019, change network for new data
        # add maxPool at stage1, and 1024 is the final conv filter number.
        # log: log_ResAttention_CV0_20190810_144024.txt
        #
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(231, 32, nGroups=33, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 64, nGroups=8, poolingLayer=None)
                        )  # ouput size: 64*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 128, nGroups=16, poolingLayer=None)
                        ) # ouput size: 128*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None)
                        ) # output size: 256*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None)
                        )  # output size: 512*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*16*16
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*8*8
        self.m_layerBeforeFc=nn.Conv2d(1024, 1024, kernel_size=8, stride=8, padding=0, bias=False)
                            # nn.MaxPool2d(4)
        self.m_fc1    = nn.Linear(1024, 1, bias=False)  # for sigmoid output, one number
        
        
        """


        """
        # For input image size: 140*251*251 (zyx)
        # at July 30 15:20, 2019, continue the refine network: 90 million parameters
        # add maxPool at stage1, and 1024 is the final conv filter number.
        # log:  log_ResAttention_CV1_20190730_152655.txt
        #       log_ResAttention_CV0_20190730_152537.txt
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 32, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 64, nGroups=8, poolingLayer=None)
                        )  # ouput size: 64*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 128, nGroups=16, poolingLayer=None)
                        ) # ouput size: 128*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None)
                        ) # output size: 256*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None)
                        )  # output size: 512*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*16*16
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*8*8
        self.m_layerBeforeFc=nn.Conv2d(1024, 1024, kernel_size=8, stride=8, padding=0, bias=False)
                            # nn.MaxPool2d(4)
        self.m_fc1    = nn.Linear(1024, 1, bias=False)  # for sigmoid output, one number
        
        """


        """
        # For input image size: 140*251*251 (zyx)
        # at July 30 10:00, 2019, continue the refine network, 294 million parameters
        # add maxPool at stage1, and 2048 is final filter number.
        # log: log_ResAttention_CV0_20190730_102840.txt
        #      log_ResAttention_CV1_20190730_103008.txt
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 32, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 64, nGroups=8, poolingLayer=None)
                        )  # ouput size: 64*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=nn.MaxPool2d(3,stride=2, padding=1)),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 128, nGroups=16, poolingLayer=None)
                        ) # ouput size: 128*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None)
                        ) # output size: 256*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None)
                        )  # output size: 512*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*16*16
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(1024, 2048, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*8*8
        self.m_layerBeforeFc=nn.Conv2d(2048, 2048, kernel_size=8, stride=8, padding=0, bias=False)
                            # nn.MaxPool2d(4)
        self.m_fc1    = nn.Linear(2048, 1, bias=False)  # for sigmoid output, one number
        
        
        """


        """
        # For input image size: 140*251*251 (zyx)
        # at July 30 03:00, 2019, continue to reduce network parameters again, 90M parameters.
        # use convStride 2, and cancel maxpool;
        # log:log_ResAttention_CV0_20190730_033913.txt, log_ResAttention_CV1_20190730_034528.txt
        #     log_ResAttention_CV2_20190730_034553.txt, log_ResAttention_CV3_20190730_034647.txt
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 32, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 64, nGroups=8, poolingLayer=None)
                        )  # ouput size: 64*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None, convStride=2),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 128, nGroups=16, poolingLayer=None)
                        ) # ouput size: 128*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(128, 128, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(128, 256, nGroups=32, poolingLayer=None)
                        ) # output size: 256*63*63
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(256, 256, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(256, 512, nGroups=32, poolingLayer=None)
                        )  # output size: 512*32*32
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(512, 512, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(512, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*16*16
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None, convStride=2),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None),
                        ResNeXtBlock(1024, 1024, nGroups=32, poolingLayer=None)
                        )  # output size: 1024*8*8
        self.m_layerBeforeFc=nn.Conv2d(1024, 1024, kernel_size=8, convStride=8, padding=0, bias=False)
                            # nn.MaxPool2d(4)
        self.m_fc1    = nn.Linear(1024, 1, bias=False)  # for sigmoid output, one number

        
        """


        """
        # For input image size: 140*251*251 (zyx)
        # at July 29 14:39, 2019, continue to reduce network parameters again,
        # use convStride 3, and cancel maxpool;
        # log:log_ResAttention_CV1_20190729_164118.txt, and  log_ResAttention_CV0_20190729_161232.txt
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 32, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 48, nGroups=8, poolingLayer=None)
                        )  # ouput size: 48*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None, convStride=3),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None),
                        ResNeXtBlock(48, 64, nGroups=12, poolingLayer=None)
                        ) # ouput size: 64*84*84
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None, convStride=3),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 80, nGroups=16, poolingLayer=None)
                        ) # output size: 80*28*28
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(80, 80, nGroups=20, poolingLayer=None, convStride=3),
                        ResNeXtBlock(80, 80, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(80, 96, nGroups=20, poolingLayer=None)
                        )  # output size: 96*10*10
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None, convStride=3),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 112, nGroups=24, poolingLayer=None)
                        )  # output size: 112*4*4
        self.m_layerBeforeFc=nn.Conv2d(112, 112, kernel_size=4, convStride=4, padding=0, bias=False)
                            # nn.MaxPool2d(4)
        self.m_fc1    = nn.Linear(112, 1, bias=False)  # for sigmoid output, one number
        
        """

        """
        # For input image size: 140*251*251 (zyx)
        # at July 29 12:45, 2019, continue to reduce network parameters again, to 247K
        # use average pooling. log:log_ResAttention_CV1_20190729_132917.txt
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 32, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(32, 32, nGroups=8, poolingLayer=None),
                        ResNeXtBlock(32, 48, nGroups=8, poolingLayer=None)
                        )  # ouput size: 48*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=nn.MaxPool2d(3)),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None),
                        ResNeXtBlock(48, 64, nGroups=12, poolingLayer=None)
                        ) # ouput size: 64*83*83
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=nn.MaxPool2d(3)),
                        ResNeXtBlock(64, 64, nGroups=16, poolingLayer=None),
                        ResNeXtBlock(64, 80, nGroups=16, poolingLayer=None)
                        ) # output size: 80*27*27
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(80, 80, nGroups=20, poolingLayer=nn.MaxPool2d(3)),
                        ResNeXtBlock(80, 80, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(80, 96, nGroups=20, poolingLayer=None)
                        )  # output size: 96*9*9
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=nn.MaxPool2d(2)),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 112, nGroups=24, poolingLayer=None)
                        )  # output size: 112*4*4
        self.m_layerBeforeFc= nn.MaxPool2d(4)
        self.m_fc1    = nn.Linear(112, 1, bias=False)  # for sigmoid output, one number
        
        
        
        """


        """
         # For input image size: 140*251*251 (zyx)
        # at July 29 11:10, 2019, continue to reduce network parameters again from  505K parameters to 155K
        # use average pooling. log: log_ResAttention_CV0_20190729_1134.txt
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 48, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None)
                        )  # ouput size: 48*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None)
                        ) # ouput size: 48*125*125
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None)
                        ) # output size: 48*62*62
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None)
                        )  # output size: 48*31*31
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None)
                        )  # output size: 48*15*15
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None),
                        ResNeXtBlock(48, 48, nGroups=12, poolingLayer=None)
                        )  # output size: 48*7*7
        self.m_layerBeforeFc= nn.AvgPool2d(7)
        self.m_fc1    = nn.Linear(48, 1, bias=False)  # for sigmoid output, one number
        
        
        """


        """
        # For input image size: 140*251*251 (zyx)
        # at July 29 09:25, 2019, continue to reduce network parameters again from  1.23 million parameters to 505K.
        # use average pooling. log: log_ResAttention_CV0_20190729_0941.txt
        self.m_stage0 = nn.Sequential(
                        ResNeXtBlock(140, 96, nGroups=20, poolingLayer=None),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None)
                        )  # ouput size: 96*251*251
        self.m_stage1 = nn.Sequential(
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None)
                        ) # ouput size: 96*126*126
        self.m_stage2 = nn.Sequential(
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None)
                        ) # output size: 96*128*128
        self.m_stage3 = nn.Sequential(
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None)
                        )  # output size: 96*64*64
        self.m_stage4 = nn.Sequential(
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None)
                        )  # output size: 96*24*24
        self.m_stage5 = nn.Sequential(
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None)
                        )  # output size: 96*16*16
        self.m_stage6 = nn.Sequential(
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=nn.AvgPool2d(2)),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None),
                        ResNeXtBlock(96, 96, nGroups=24, poolingLayer=None)
                        )  # output size: 96*8*8
        self.m_layerBeforeFc= nn.AvgPool2d(8)
        self.m_fc1    = nn.Linear(96, 1, bias=False)  # for sigmoid output, one number

        
        """    


        """
        # For input image size: 140*251*251 (zyx)
        # at July 27 16:20, 2019, reduce network parameters again from  1.23 million parameters to
        # use average pooling. log_ResAttention_CV0_20190727_1708.txt
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
        self.m_layerBeforeFc= nn.AvgPool2d(8)
        self.m_fc1    = nn.Linear(160, 1, bias=False)  # for sigmoid output, one number
        
        """


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
        self.m_layerBeforeFc= nn.AvgPool2d(8)
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
        self.m_layerBeforeFc= nn.AvgPool2d(8)
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
        self.m_layerBeforeFc= nn.AvgPool2d(8)
        self.m_fc1    = nn.Linear(2048, 1, bias=False)  # for sigmoid output, one number
        
        """


