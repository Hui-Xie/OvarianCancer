import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
from torch._jit_internal import weak_module, weak_script_method
import torch
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
import numpy as np
import sys


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
class BoundaryLoss1(_Loss):
    """
    Boundary Loss, please refer paper: Boundary Loss for highly Unbalanced Segmentation, in link: https://arxiv.org/abs/1812.07032
    outside boundary of ground truth, it is positive distance, a penalty to increase loss;
    inside  boundary of ground truth, it is negative distance, a reward to reduce loss;
    Support K classes classification.
    support 2D and 3D images.
    """
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=0.001, k=2, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda=lambdaCoeff # weight coefficient of whole loss function
        self.m_k = k              # k classes classification, m_k=2 is for binary classification, etc
        self.m_weight = torch.ones(self.m_k) if weight is None else weight
        if len(self.m_weight) != self.m_k:
            print(f"Error: the number of classes does not match weight in the Boundary Loss init method")
            sys.exit(-5)


    @weak_script_method
    def forward(self, inputx, target):
        softmaxInput = F.softmax(inputx, 1)
        targetNumpy = target.cpu().numpy().astype(int)
        shape = targetNumpy.shape
        ndim = targetNumpy.ndim
        N = shape[0]     # batch Size
        dilateFilter = np.ones((3,)*(ndim-1), dtype=int)  # dilation filter for for 4-connected boundary
        ret = torch.zeros(N).cuda()

        for k in range(1,self.m_k):  # ignore background with k starting with 1
            segProb = torch.narrow(softmaxInput,1, k,1)
            segProb = torch.squeeze(segProb, 1)

            targetk = (targetNumpy == k)
            targetkNot = (targetNumpy != k)
            levelSet = np.zeros(shape)

            for i in range(N):
                if np.count_nonzero(targetk[i]) == 0:
                    continue
                boundary = binary_dilation(targetkNot[i],dilateFilter) & targetk[i]
                inside = targetk[i] ^ boundary  # xor operator
                signMatrix = inside*(-1)+ targetkNot[i]
                levelSet[i] = ndimage.distance_transform_edt(boundary==0)*signMatrix

            levelSetTensor = torch.from_numpy(levelSet).float().cuda()
            x = torch.mean(segProb * levelSetTensor, dim=tuple([i for i in range(1,ndim)]))
            x = torch.squeeze(x)
            ret += x*self.m_weight[k]

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret*self.m_lambda


@weak_module
class BoundaryLoss2(_Loss):
    """
    this is an improved version of Boundary Loss.
    It idea is improving from the paper: Boundary Loss for highly Unbalanced Segmentation, in link: https://arxiv.org/abs/1812.07032
    This improved version has no negative distance in the level set.

    Loss function only cares about the error segmentation, and ignore correct both forground  and background segmentations.

    Here, A indicates ground truth "1" excluding the intersection C with prediction "1". A is the wanting part.
          B indicates predicted segmentation "1" excluding the intersection C. B is the leaky part.
          C indicate the overlapped  intersection of ground truth 1  and prediction 1. C is the correctly segmented part.
          A+C = ground truth 1; B+C = prediction 1.

    In other words, we only care about loss on both A and B, the wanting and leaky part,  and  ignoring all others.

    loss = SegProb * levelSetB + (1-SegProb)*levelSetA,
         where levelSetB means the distance map from only B to C;
               levelSetA means the distance map from only A to C;
               when C=NUll, there are speicial case detailed in code blow.

    The perfect goal of this loss optimization is to make A=null and B =Null, at same time C !=Null.

    Support K classes classification.
    support 2D and 3D images.
    """
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, k=2, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda=lambdaCoeff # weight coefficient of whole loss function
        self.m_k = k              # k classes classification, m_k=2 is for binary classification, etc
        self.m_weight = torch.ones(self.m_k) if weight is None else weight
        if len(self.m_weight) != self.m_k:
            print(f"Error: the number of classes does not match weight in the Boundary Loss init method")
            sys.exit(-5)


    @weak_script_method
    def forward(self, inputx, target):
        # case1: ACB (this is the frequent case; and in some case, A \subset C,  or B \subset C, and C is not Null.
        # case2: AB (there is no overlap between ground truth 1 and prediction 1; in other words, C=Null.)
        # Case3: A (there is only ground truth 1, but no prediciton 1, and C=Null)
        # case4: B (there is only prediction 1, but no groundtruth 1, and C=Null)
        # case5: Null (there is no prediction 1 and groundtruth 1, so no loss at all)

        prediction = torch.argmax(inputx, dim=1)

        softmaxInput = F.softmax(inputx, 1)
        targetNumpy = target.cpu().numpy().astype(int)
        shape = targetNumpy.shape
        ndim = targetNumpy.ndim
        N = shape[0]     # batch Size
        ret = torch.zeros(N).cuda()

        for k in range(1,self.m_k):  # ignore background with k starting with 1
            segProb = torch.narrow(softmaxInput,1, k,1)
            segProb = torch.squeeze(segProb, 1)

            targetk = (targetNumpy == k)
            predictionk = (prediction == k)

            levelSetA = np.zeros(shape)  # default Loss = 0 for A and B
            levelSetB = np.zeros(shape)

            # for the A,B,C, they are needed in the context of each sample
            for i in range(N):
                # if np.count_nonzero(targetk[i,]) == 0:
                #     continue
                C = targetk[i,] * predictionk[i,]
                A = targetk[i,] ^ C
                B = predictionk[i,] ^ C

                # case1: ACB (this is the frequent case; and in some case, A \subset C,  or B \subset C, and C is not Null.
                if np.count_nonzero(C) != 0:
                    CNot = np.invert(C)
                    levelSetC = ndimage.distance_transform_edt(CNot)
                    levelSetA[i,] = A*levelSetC  # distance AC is bigger than distance AB, as C \subset B.
                    levelSetB[i,] = B*levelSetC

                # case2: AB (there is no overlap between ground truth 1 and prediction 1; in other words, C=Null.)
                elif np.count_nonzero(A) >0 and np.count_nonzero(B) >0:
                    # when A is farther to B, we hope to get bigger gradient on pixels of A.
                    levelSetA[i,] = A * ndimage.distance_transform_edt(np.invert(B))
                    levelSetB[i,] = B * ndimage.distance_transform_edt(np.invert(A))

                 # Case3: A (there is only ground truth 1, but no prediciton 1, and C=Null)
                elif np.count_nonzero(A) >0:
                    levelSetA[i,] = A

                # case4: B (there is only prediction 1, but no groundtruth 1, and C=Null)
                elif np.count_nonzero(B) >0:
                    levelSetB[i,] = B

                # case5: Null (there is no prediction 1 and groundtruth 1, so no loss at all)
                else:
                    continue


            levelSetATensor = torch.from_numpy(levelSetA).float().cuda()
            levelSetBTensor = torch.from_numpy(levelSetB).float().cuda()
            x = torch.mean(segProb * levelSetBTensor+ (1-segProb)*levelSetATensor, dim=tuple([i for i in range(1,ndim)]))
            x = torch.squeeze(x)
            ret += x*self.m_weight[k]

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret*self.m_lambda