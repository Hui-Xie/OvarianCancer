from BasicModel import BasicModel
from BuildingBlocks import *
import torch

# SkyWatcher Model, simultaneously train segmentation and treatment response

class SkyWatcherModel(BasicModel):
    def __init__(self):
        super().__init__()

    def encoderForward(self, inputx):
        x = self.m_input(inputx)
        for down in self.m_downList:
            x = down(x)
        # here x is the output at crossing point of sky watcher
        return x

    def responseForward(self, crossingx):
        # xr means x rightside output, or response output
        xr = crossingx
        xr = self.m_11Conv(xr)
        xr = torch.reshape(xr, (xr.shape[0], xr.numel()//xr.shape[0]))
        xr = self.m_fc11(xr)
        # # ===debug===
        # print("before Fully connect layers:")
        # print(xr)
        # for module in self.m_fc11._modules.values():
        #     xr = module(xr)
        #     print(module.__class__.__name__ )
        #     print(xr)
        # # ===debug===
        return xr

    def decoderForward(self, crossingx):
        # xup means the output using upList
        xup = crossingx
        for up in self.m_upList:
            xup = up(xup)
        xup = self.m_upOutput(xup)
        return xup


    def forward(self, inputx, bPurePrediction=False):
        x = self.encoderForward(inputx)
        xr = self.responseForward(x)
        if bPurePrediction:
            return xr
        else:
            xup = self.decoderForward(x)
            return xr, xup

    @staticmethod
    def freezeModuleList(moduleList, requires_grad=False):
        for module in moduleList:
            for param in module.parameters():
                param.requires_grad = requires_grad

    def freezeResponseBranch(self, requires_grad=False):
        moduleList = [self.m_11conv, self.m_fc11]
        self.freezeModuleList(moduleList, requires_grad= requires_grad)

    def freezeSegmentationBranch(self, requires_grad=False):
        self.freezeEncoder(requires_grad=requires_grad)
        self.freezeDecoder(requires_grad=requires_grad)

    def freezeEncoder(self, requires_grad=False):
        moduleList = [self.m_input, self.m_downList]
        self.freezeModuleList(moduleList, requires_grad=requires_grad)

    def freezeDecoder(self, requires_grad=False):
        moduleList = [self.m_upList, self.m_upOutput]
        self.freezeModuleList(moduleList, requires_grad=requires_grad)
