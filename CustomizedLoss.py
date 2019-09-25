import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
import numpy as np
import sys



class FocalCELoss(_WeightedLoss):
    """
    Focal Loss, please refer paper: "Focal Loss for Dense Object Detection" in link: https://arxiv.org/abs/1708.02002
    """
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, gamma =2.0, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self.gamma = gamma
        self.ignore_index = ignore_index


    def forward(self, inputx, target):
        focalFactor = (1 - F.softmax(inputx, 1)) ** self.gamma
        return F.nll_loss(focalFactor * F.log_softmax(inputx, 1), target, self.weight, None, self.ignore_index, None, self.reduction)

    def setGamma(self,gamma):
        self.gamma = gamma



class BoundaryLoss1(_Loss):
    """
    Boundary Loss, please refer paper: Boundary Loss for highly Unbalanced Segmentation, in link: https://arxiv.org/abs/1812.07032
    outside boundary of ground truth, it is positive distance, a penalty to increase loss;
    inside  boundary of ground truth, it is negative distance, a reward to reduce loss;
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

    def forward(self, inputx, target):
        inputxMaxDim1, _= torch.max(inputx, dim=1, keepdim=True)
        inputxMaxDim1 = inputxMaxDim1.expand_as(inputx)
        softmaxInput = F.softmax(inputx-inputxMaxDim1, 1)  #use inputMaxDim1 is to avoid overflow.

        targetNumpy = target.cpu().numpy().astype(int)
        shape = targetNumpy.shape
        ndim = targetNumpy.ndim
        N = shape[0]     # batch Size
        dilateFilter = np.ones((3,)*(ndim-1), dtype=int)  # dilation filter for for 4-connected boundary in 2D, 8 connected boundary in 3D
        ret = torch.zeros(N).to(inputx.device)

        for k in range(1,self.m_k):  # ignore background with k starting with 1
            segProb = torch.narrow(softmaxInput,1, k,1)
            segProb = torch.squeeze(segProb, 1)

            targetk = (targetNumpy == k)
            targetkNot = (targetNumpy != k)
            levelSet = np.zeros(shape)

            for i in range(N):
                if np.count_nonzero(targetk[i]) == 0:
                    levelSet[i].fill(1)
                else:
                    boundary = binary_dilation(targetkNot[i],dilateFilter) & targetk[i]
                    inside = targetk[i] ^ boundary  # xor operator
                    signMatrix = inside*(-1)+ targetkNot[i]
                    levelSet[i] = ndimage.distance_transform_edt(boundary==0)*signMatrix

            levelSetTensor = torch.from_numpy(levelSet).float().to(inputx.device)
            x = torch.mean(segProb * levelSetTensor, dim=tuple([i for i in range(1,ndim)]))
            x = torch.squeeze(x)
            ret += x*self.m_weight[k]

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret*self.m_lambda

class BoundaryLoss2(_Loss):
    """
    this is an improved version of Boundary Loss.
    It idea is improving from the paper: Boundary Loss for highly Unbalanced Segmentation, in link: https://arxiv.org/abs/1812.07032
    This improved version has no negative distance in the level set, and it uses the initial prediction and ground truth together to decide levelset,
         comparing with original paper.


    Loss function only cares about the error segmentation, and ignore correct both foreground  and background segmentations.

    Here, A indicates ground truth "1" excluding the intersection C with prediction "1". A is the wanting part or undersegmented part.
          B indicates predicted segmentation "1" excluding the intersection C. B is the leaky part.
          C indicate the overlapped  intersection of ground truth 1  and prediction 1. C is the correctly segmented part.
          A+C = ground truth 1; B+C = prediction 1.

    In other words, we only care about loss on both A and B, the wanting and leaky part,  and  ignoring all others.

    loss = -log(1-SegProb) * levelSetB - log(SegProb)*levelSetA,
         where levelSetB means the distance map from only B to C;
               levelSetA means the distance map from only A to C;
               when C=NUll, there are special cases detailed in code blow.
         using log is to make loss at same quantity level comparing with cross entropy.


    The perfect goal of this loss optimization is to make A=null and B =Null, at same time C !=Null.

    Support K classes classification.
    support 2D and 3D images.
    """
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, k=2, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda=lambdaCoeff # weight coefficient of whole loss function
        self.m_k = k              # k classes classification, m_k=2 is for binary classification, etc
        self.weight = torch.ones(self.m_k) if weight is None else weight  # keep name consistent with CrossEntropy
        if len(self.weight) != self.m_k:
            print(f"Error: the number of classes does not match weight in the Boundary Loss init method")
            sys.exit(-5)



    def forward(self, inputx, target):
        # case1: ACB (this is the frequent case; and in some case, A \subset C,  or B \subset C, and C is not Null.
        # case2: AB (there is no overlap between ground truth 1 and prediction 1; in other words, C=Null.)
        # Case3: A (there is only ground truth 1, but no prediciton 1, and C=Null)
        # case4: B (there is only prediction 1, but no groundtruth 1, and C=Null)
        # case5: Null (there is no prediction 1 and groundtruth 1, so no loss at all)

        prediction = torch.argmax(inputx, dim=1).cpu().numpy()

        # softmax(x) = softmax(x+c) where c is scalar. subtracting max(x) avoids overflow of exponential explosion.
        # softmaxInput = F.softmax(inputx-inputx.max(), 1)
        assert self.m_k == 2
        logsoftmax = F.log_softmax(inputx, dim=1)  # use logsoftmax to avoid overflow and underflow.


        targetNumpy = target.cpu().numpy().astype(int)
        shape = targetNumpy.shape
        ndim = targetNumpy.ndim
        N = shape[0]     # batch Size
        ret = torch.zeros(N).to(inputx.device)


        for k in range(1,self.m_k):  # ignore background with k starting with 1
            logP = torch.narrow(logsoftmax,1, k,1)
            log1_P = torch.narrow(logsoftmax,1, 0,1)

            logP = torch.squeeze(logP, 1)
            log1_P = torch.squeeze(log1_P,1)

            targetk = (targetNumpy == k)
            predictionk = (prediction == k)

            levelSetA = np.zeros(shape)  # default Loss = 0 for A and B
            levelSetB = np.zeros(shape)

            ABSize = torch.zeros(N, requires_grad=False,device=inputx.device)

            # for the A,B,C, they are needed in the context of each sample
            for i in range(N):
                # if np.count_nonzero(targetk[i,]) == 0:
                #     continueA
                C = targetk[i,] * predictionk[i,]
                A = targetk[i,] ^ C
                B = predictionk[i,] ^ C
                ABSize[i] = np.count_nonzero(A) + np.count_nonzero(B)

                # case1: ACB (this is the frequent case; and in some case, A \subset C,  or B \subset C, and C is not Null.
                if np.count_nonzero(C) != 0:
                    CNot = np.invert(C)
                    levelSetC = ndimage.distance_transform_edt(CNot)
                    levelSetA[i,] = A*levelSetC  # distance AC is bigger than distance AB, as C \subset B.
                    levelSetB[i,] = B*levelSetC

                # case2: AB (there is no overlap between ground truth 1 and prediction 1; in other words, C=Null.)
                elif np.count_nonzero(A) >0 and np.count_nonzero(B) >0:
                    # when no-overlapping A is farther from B, we hope to get bigger gradient on pixels of A.
                    levelSetA[i,] = A * ndimage.distance_transform_edt(np.invert(B))
                    levelSetB[i,] = B * ndimage.distance_transform_edt(np.invert(A))

                 # Case3: A (there is only ground truth 1, but no prediciton 1, and C=Null)
                 #        in this case, boundary loss degrades into general cross entropy loss .
                elif np.count_nonzero(A) >0:
                    levelSetA[i,] = A

                # case4: B (there is only prediction 1, but no groundtruth 1, and C=Null)
                elif np.count_nonzero(B) >0:
                    levelSetB[i,] = B

                # case5: Null (there is no prediction 1 and groundtruth 1, so no loss at all)
                else:
                    continue


            levelSetATensor = torch.from_numpy(levelSetA).float().to(inputx.device)
            levelSetBTensor = torch.from_numpy(levelSetB).float().to(inputx.device)
            x = torch.sum(-log1_P * levelSetBTensor - logP*self.weight[k]*levelSetATensor, dim=tuple([i for i in range(1,ndim)]))
            x = torch.squeeze(x)
            x /= ABSize +1e-8    #default 1e-8 is to avoid divided by 0.
            ret += x

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret*self.m_lambda



class BoundaryLoss3(_Loss):
    """
    this is an improved version of Boundary Loss.
    It idea is improving from the paper: Boundary Loss for highly Unbalanced Segmentation, in link: https://arxiv.org/abs/1812.07032
    This improved version has no negative distance in the level set, and it uses ground truth only to decide levelset.

    CrossEntropy is indiscriminately to treat all pixels, whatever it  is far away from groundtruth or in the middle of groundtruth.

    The easier judging pixels get higher distance weight, which means they get higher gradient, and quickly converge to expected goal.

    loss = -log(SegProb) * levelSetFg - log(1-SegProb)*levelSetBg,
         where levelSetFg means the distance map from foreground  to background;
               levelSetBg means the distance map from background  to foreground;
               SegProb is the predicting probability of foreground.


    The perfect goal of this loss optimization is to make loss= 0

    Support K classes classification.
    support 2D and 3D images.
    """
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, k=2, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda=lambdaCoeff # weight coefficient of whole loss function
        self.m_k = k              # k classes classification, m_k=2 is for binary classification, etc
        self.weight = torch.ones(self.m_k) if weight is None else weight  # keep name consistent with CrossEntropy
        if len(self.weight) != self.m_k:
            print(f"Error: the number of classes does not match weight in the Boundary Loss init method")
            sys.exit(-5)



    def forward(self, inputx, target):
        assert self.m_k == 2
        logsoftmax = F.log_softmax(inputx, dim=1)  # use logsoftmax to avoid overflow and underflow.

        targetNumpy = target.cpu().numpy().astype(int)
        shape = targetNumpy.shape
        ndim = targetNumpy.ndim
        N = shape[0]     # batch Size
        ret = torch.zeros(N).to(inputx.device)

        for k in range(1,self.m_k):  # ignore background with k starting with 1
            logP = torch.narrow(logsoftmax,1, k,1)
            log1_P = torch.narrow(logsoftmax,1, 0,1)

            logP = torch.squeeze(logP, 1)
            log1_P = torch.squeeze(log1_P,1)

            targetk = (targetNumpy == k)

            levelSetFg = np.zeros(shape)
            levelSetBg = np.zeros(shape)

            for i in range(N):
                if np.count_nonzero(targetk[i,]) == 0:
                    levelSetBg[i,].fill_(1)
                else:
                    Fg = targetk[i,]
                    Bg = ~Fg
                    levelSetFg[i,] = ndimage.distance_transform_edt(Fg)
                    levelSetBg[i,] = ndimage.distance_transform_edt(Bg)

            levelSetFgTensor = torch.from_numpy(levelSetFg).float().to(inputx.device)
            levelSetBgTensor = torch.from_numpy(levelSetBg).float().to(inputx.device)
            # x = torch.mean(-logP *self.weight[k]* levelSetFgTensor - log1_P*levelSetBgTensor, dim=tuple([i for i in range(1,ndim)]))
            x = torch.mean(-(logP*self.weight[k]-log1_P) * (levelSetFgTensor - levelSetBgTensor), dim=tuple([i for i in range(1, ndim)]))
            x = torch.squeeze(x)
            ret += x

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret*self.m_lambda
