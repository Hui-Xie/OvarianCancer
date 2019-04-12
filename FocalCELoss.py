import warnings

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch._jit_internal import weak_module, weak_script_method


@weak_module
class FocalCELoss(_WeightedLoss):
    '''
    Focal Loss, please refer paper: "Focal Loss for Dense Object Detection" in link: https://arxiv.org/abs/1708.02002
    '''
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, gamma =2.0, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    @weak_script_method
    def forward(self, input, target):
        focalFactor = (1- F.softmax(input, 1))**self.gamma
        return F.nll_loss(focalFactor*F.log_softmax(input, 1), target, self.weight, None, self.ignore_index, None, self.reduction)

    def setGamma(self,gamma):
        self.gamma = gamma
