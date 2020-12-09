import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
import torch
import torch.nn as nn
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
import numpy as np
import sys

def logits2Prob(x, dim):
    # convert logits to probability for input x
    xMaxDim, _ = torch.max(x, dim=dim, keepdim=True)
    xMaxDim = xMaxDim.expand_as(x)
    prob = F.softmax(x - xMaxDim, dim=dim)  # using inputMaxDim is to avoid overflow.
    return prob


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
        # Todo: make dim=1 is correct in inputx dimenson.
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


class FullCrossEntropyLoss(_Loss):
    """

    """
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, k=2, weight=None, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda=lambdaCoeff # weight coefficient of whole loss function
        self.m_k = k              # k classes classification, m_k=2 is for binary classification, etc
        self.weight = torch.ones(self.m_k) if weight is None else weight  # keep name consistent with CrossEntropy
        if len(self.weight) != self.m_k:
            print(f"Error: the number of classes does not match weight in the Loss init method")
            sys.exit(-5)



    def forward(self, inputx, target):
        assert self.m_k == 2
        logsoftmax = F.log_softmax(inputx, dim=1)  # use logsoftmax to avoid overflow and underflow.

        ndim = target.ndim

        logP = torch.narrow(logsoftmax,1, 1,1)
        log1_P = torch.narrow(logsoftmax,1, 0,1)

        logP = torch.squeeze(logP, 1)
        log1_P = torch.squeeze(log1_P,1)

        targetFloat = target.type(torch.float)

        x = torch.mean(-(logP*self.weight[1]-log1_P) * (targetFloat*2.0-1.0), dim=tuple([i for i in range(1, ndim)]))
        x = torch.squeeze(x)
        ret = x

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret*self.m_lambda

class DistanceCrossEntropyLoss(_Loss):
    """
    in loss functino design, putting  penalty and reward together will confuse newtwork, as for same neuro position,
    different input samples may get penalty or reward at this neuro position. Pure penalty will make one nero get singel
    gradient change diretion.

    """
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, k=2, weight=None, size_average=None, reduce=None, reduction='mean', trancateDistance=5):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda=lambdaCoeff # weight coefficient of whole loss function
        self.m_k = k              # k classes classification, m_k=2 is for binary classification, etc
        self.weight = torch.ones(self.m_k) if weight is None else weight  # keep name consistent with CrossEntropy
        if len(self.weight) != self.m_k:
            print(f"Error: the number of classes does not match weight in the Loss init method")
            sys.exit(-5)
        self.m_trancateDistance = trancateDistance

    def forward(self, inputx, target):
        assert self.m_k == 2
        logsoftmax = F.log_softmax(inputx, dim=1)  # use logsoftmax to avoid overflow and underflow.

        targetNumpy = target.cpu().numpy().astype(int)
        shape = targetNumpy.shape
        ndim = targetNumpy.ndim
        N = shape[0]  # batch Size
        ret = torch.zeros(N).to(inputx.device)

        for k in range(1, self.m_k):  # ignore background with k starting with 1
            logP = torch.narrow(logsoftmax, 1, k, 1)
            log1_P = torch.narrow(logsoftmax, 1, 0, 1)

            logP = torch.squeeze(logP, 1)
            log1_P = torch.squeeze(log1_P, 1)

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
                    levelSetFg[i,] = np.clip(levelSetFg[i,], 0, self.m_trancateDistance)
                    levelSetBg[i,] = np.clip(levelSetBg[i,], 0, self.m_trancateDistance)

            levelSetFgTensor = torch.from_numpy(levelSetFg).float().to(inputx.device)
            levelSetBgTensor = torch.from_numpy(levelSetBg).float().to(inputx.device)
            x = torch.mean(-logP * self.weight[k] *levelSetFgTensor - log1_P *levelSetBgTensor,  dim=tuple([i for i in range(1, ndim)]))
            x = torch.squeeze(x)
            ret += x

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret * self.m_lambda


class GeneralizedBinaryDiceLoss(_Loss):
    """
     please refer paper "Generalized Dice overlap as a deep learning loss function for highly unbalanced segmentations"
     at link: https://arxiv.org/pdf/1707.03237.pdf

     Current only support binary segmentation.

    """
    __constants__ = ['reduction']

    def __init__(self, lambdaCoeff=1, size_average=None, reduce=None, reduction='mean'):
        super().__init__(size_average, reduce, reduction)
        self.m_lambda=lambdaCoeff # weight coefficient of whole loss function

    def forward(self, inputx, target):
        inputxMaxDim1, _= torch.max(inputx, dim=1, keepdim=True)
        inputxMaxDim1 = inputxMaxDim1.expand_as(inputx)
        softmaxInput = F.softmax(inputx-inputxMaxDim1, 1)  #use inputMaxDim1 is to avoid overflow.

        batchSize = target.shape[0]
        N = target.numel()/batchSize
        sampleDims = tuple(range(1, target.dim()))

        # all variables below include batch dimension
        P = torch.narrow(softmaxInput, dim=1,start=1,length=1)
        P = torch.squeeze(P, dim=1)
        T = target.type(torch.float32)
        N = torch.ones((batchSize,1), device=inputx.device, dtype=torch.float32)*N
        N1 = torch.sum(T, dim=sampleDims).type(torch.float32)
        N2 = N-N1
        w1 =  1.0/(N1*N1)
        w2 =  1.0/(N2*N2)

        PPlusT = torch.sum(P+T, dim=sampleDims)
        PTimesT = torch.sum(P*T, dim=sampleDims)
        numerator = (w1+w2)*PTimesT-w2*PPlusT+N*w2  # use 1-P1 to replace P2.
        denominator = (w1-w2)*PPlusT + 2.0*N*w2
        ret = 1.0-2.0*numerator/denominator  # GDL in batch

        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)
        return ret*self.m_lambda

# support multiclass generalized Dice Loss
class GeneralizedDiceLoss():
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target):
        '''
         target: for N surfaces, surface 0 in its exact location marks as 1, surface N-1 in its exact location marks as N.
                region pixel between surface i and surface i+1 marks as i.

        :param inputx: float32 tensor, size of (B,K,H,W), where K is the number of classes. inputx is softmax probability along dimension K.
        :param target: long tensor, size of (B,H,W), where each element has a long value of [0,K) indicate the belonging class
        :return: a float scalar of mean dice over all classes and over batchSize.

        '''
        B,K,H,W = inputx.shape
        assert (B,H,W) == target.shape
        assert K == target.max()+1
        device = inputx.device

        # convert target of size(B,H,W) into (B,K,H,W) into one-hot float32 probability
        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device)
        for k in range(0,K):
            targetProb[:,k,:,:] = torch.where(k ==target, torch.ones_like(target), targetProb[:,k,:,:])

        # compute weight of each class
        W = torch.zeros(K, device=device, dtype=torch.float64)
        for k in range(0,K):
            W[k] = torch.tensor(1.0).double()/((k == target).sum()**2)

        # generalized dice loss
        sumDims = (0,2,3)
        GDL = 1.0-2.0*((inputx*targetProb).sum(dim=sumDims)*W).sum()/((inputx+targetProb).sum(dim=sumDims)*W).sum()
        return GDL

class SmoothSurfaceLoss():
    def __init__(self, mseLossWeight=10.0):
        self.mseLossWeight = mseLossWeight  # weight for embedded MSE loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target):
        '''
        measure  a sum of the square mean errors between S0-S1 and G0-G1 along H and W dimension respectively.

        Smooth Loss is a Mean Square Error measure between (S0 -S1) and (G0-G1),
        where S0  and G0  are 0 to N-1 surfaces' prediction and ground truth, and S1 and G1  are 1 to N surfaces prediction and ground truth.
        In other words, we expect the distance difference between predicted surface 0  and surface 1
        are consistent with the distance difference between ground truth surface 0 and surface 1.

        This loss forces network to learn the distance difference between surfaces, the layer width.
        For example, if predicted surface 0 deviates ground truth surface 0 5 pixels,
        this loss will forces  the predicted surface 1 also deviates ground truth surface 1 5 pixels, otherwise it will get some loss.
        This is along height direction shift.

        Similarly, in width direction, we compute similar MSE along width direction shift.
        Its goal is to learn surface curve changes along width direction.

        Smooth Loss forces predicted surface waveShape / layerWidth  to be similar with ground truth.
        But it ignores surface location problem, so it needs to use with general MSE Loss together.
        General MSE Loss  gets surface locations, while SmoothLoss use relationship between surfaces
        or between adjacent columns to get better neighbour surface location through learning waveShape and layer width.
        We can also think SmoothLoss amplifies the error of MSELoss.


        :param inputx: float32 tensor surface locations, size of (B,N,W), where N is the number of surfaces.
        :param target: ground truth tensor surface locations, size of (B,N,W), where N is the number of surfaces.
        :return: a sum of square mean error between S0-S1 and G0-G1 along H and W dimension respectively.

        '''
        B,N,W = inputx.shape
        assert (B,N,W) == target.shape

        # along N dimension
        Sh0 = inputx[:,0:-1,:]  # size: B,(N-1),W
        Sh1 = inputx[:,1:,  :]
        Gh0 = target[:,0:-1,:]
        Gh1 = target[:,1:,  :]

        # along W dimension
        Sw0 = inputx[:, :, 0:-1]  # size: B,N,(W-1)
        Sw1 = inputx[:, :, 1:]
        Gw0 = target[:, :, 0:-1]
        Gw1 = target[:, :, 1:]
        loss = torch.pow((Sh0-Sh1)-(Gh0-Gh1), 2.0).mean()+torch.pow((Sw0-Sw1)-(Gw0-Gw1), 2.0).mean()\
                + self.mseLossWeight* torch.pow(inputx-target, 2.0).mean()

        return loss

class SmoothThicknessLoss():
    def __init__(self, mseLossWeight=10.0):
        self.mseLossWeight = mseLossWeight  # weight for embedded MSE loss

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target):
        '''
        Similar with SmoothSurfaceLoss, this SmoothThicknessLoss assure the wave shape change of thickness consistent
        between prediction and target.
        '''
        B,N,W = inputx.shape
        assert (B,N,W) == target.shape

        # along W dimension, use 3-point gradient formula
        Sw0 = inputx[:, :, 0:-2]  # size: B,N,(W-2)
        Sw1 = inputx[:, :, 1:-1]
        Sw2 = inputx[:, :, 2:]

        Gw0 = target[:, :, 0:-2]
        Gw1 = target[:, :, 1:-1]
        Gw2 = target[:, :, 2:]
        loss = torch.pow((Sw0+Sw2-2.0*Sw1)-(Gw0+Gw2-2.0*Gw1), 2.0).mean()\
                + self.mseLossWeight* torch.pow(inputx-target, 2.0).mean()

        return loss

# support multiclass CrossEntropy Loss
class MultiSurfaceCrossEntropyLoss():
    def __init__(self,  weight=None):
        self.m_weight = weight   # B,N,H,W, where N is the num of surfaces.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target):
        '''
        multiclass surface location cross entropy.
        this loss expect the corresponding prob at target location has maximum.

        :param inputx:  softmax probability along H dimension, in size: B,N,H,W
        :param target:  size: B,N,W; indicates surface location
        :return: a scalar
        '''
        B,N,H,W = inputx.shape
        assert (B,N,W) == target.shape
        device = inputx.device

        targetIndex = (target +0.5).long().unsqueeze(dim=-2) # size: B,N,1,W

        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device)  # size: B,N,H,W
        targetProb.scatter_(2, targetIndex, torch.ones_like(targetIndex))

        e = 1e-6
        inputx = inputx + e
        inputx = torch.where(inputx >= 1, (1 - e) * torch.ones_like(inputx), inputx)
        if self.m_weight is not None:
            loss = -(self.m_weight * (targetProb * inputx.log() + (1 - targetProb) * (1 - inputx).log())).mean()
        else:
            loss = -(targetProb * inputx.log() + (1 - targetProb) * (1 - inputx).log()).mean()
        return loss

# support multiclass CrossEntropy Loss
class MultiLayerCrossEntropyLoss():
    def __init__(self, weight=None):
        self.m_weight = weight  # B,N,H,W, where N is nLayer, instead of num of surfaces.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputx, target):
        '''
         pixel-wised multiLayer cross entropy.
        :param inputx:  N-layer probability of size: B,N,H,W
        :param target:  long tensor, size of (B,H,W), where each element has a long value of [0,N-1] indicate the belonging class
        :return: a scalar of loss
        '''
        B,N,H,W = inputx.shape
        assert (B,H,W) == target.shape
        device = inputx.device

        # convert target of size(B,H,W) into (B,N,H,W) into one-hot float32 probability
        targetProb = torch.zeros(inputx.shape, dtype=torch.long, device=device)  # size: B,N,H,W
        for k in range(0, N): # N layers
            targetProb[:, k, :, :] = torch.where(k == target, torch.ones_like(target), targetProb[:, k, :, :])

        e = 1e-6
        # 1e-8 is not ok, A=(1-e)*torch.ones_like(inputx) will still 1. and (1-A).log() will get -inf.
        inputx = inputx+e
        inputx = torch.where(inputx>=1, (1-e)*torch.ones_like(inputx), inputx)
        if self.m_weight is not None:
            loss = -(self.m_weight * (targetProb * inputx.log()+(1-targetProb)*(1-inputx).log())).mean()
        else:
            loss = -(targetProb * inputx.log()+(1-targetProb)*(1-inputx).log()).mean()
        return loss


class WeightedDivLoss():
    def __init__(self, weight=None):
        self.m_weight = weight  # B,N,H,W, where N is the num of surfaces.

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, inputxLogProb, target):
        '''
         loss = (\sum ( w_i*g_i*abs(log(g_i)-log(p_i)) ))/Num

        :param inputxLogProb:  size: B,N,H,W, log probability of prediction
        :param target:  size: B,N,H,W, ground truth probility.
        :return: a scalar of loss
        '''
        B,N,H,W = inputxLogProb.shape
        assert (B,N,H,W) == target.shape
        if self.m_weight is not None:
            assert (B,N,H,W) == self.m_weight.shape

        e = 1e-6
        # 0*np.inf= nan
        # if target=0 -> target.log()= -inf -> logG_P =inf -> loss =nan.
        # 1e-8 is not ok, A=(1-e)*torch.ones_like(inputx) will still 1. and (1-A).log() will get -inf.
        target = target + e
        target = torch.where(target >= 1, (1 - e) * torch.ones_like(target), target)

        logG_P = target.log()-inputxLogProb
        logG_P = torch.abs(logG_P)          #torch.where(logG_P >=0, logG_P, -logG_P)
        if self.m_weight is not None:
            loss = (self.m_weight * target * logG_P).mean()
        else:
            loss = (target * logG_P).mean()
        return loss


