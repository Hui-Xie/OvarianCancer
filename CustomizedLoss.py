import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
from torch._jit_internal import weak_module, weak_script_method
import torch
from scipy import ndimage
import numpy as np


@weak_module
class FocalCELoss(_WeightedLoss):
    """
    Focal Loss, please refer paper: "Focal Loss for Dense Object Detection" in link: https://arxiv.org/abs/1708.02002
    """
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, gamma =2.0, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index

    @weak_script_method
    def forward(self, inputx, target):
        focalFactor = (1 - F.softmax(inputx, 1)) ** self.gamma
        return F.nll_loss(focalFactor * F.log_softmax(inputx, 1), target, self.weight, None, self.ignore_index, None, self.reduction)

    def setGamma(self,gamma):
        self.gamma = gamma


@weak_module
class BoundaryLoss(_Loss):
    """
    Boundary Loss, please refer paper: Boundary Loss for highly Unbalanced Segmentation, in link: https://arxiv.org/abs/1812.07032
    """
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)

    @weak_script_method
    def forward(self, inputx, target):
        segProb = torch.narrow(F.softmax(inputx, 1),1, 1,1)

        targetNot = (target == 0).numpy()
        shape = targetNot.shape
        ndim = targetNot.ndim
        N = shape[0]
        levelSet = np.zeros(shape)
        for i in range(N):
            levelSet[i] = ndimage.distance_transform_edt(targetNot[i])

        levelSetTensor = torch.from_numpy(levelSet)
        ret = torch.mean(segProb * levelSetTensor, dim=tuple([i for i in range(1,ndim)]))

        return ret

