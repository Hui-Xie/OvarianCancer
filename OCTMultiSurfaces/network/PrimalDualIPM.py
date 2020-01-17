
import torch
import torch.nn as nn

class SeparationPrimalDualIPMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Mu, Sigma2Reciprocal, A, S0, Lamda, alpha, epsilon):
        '''
        S* = arg min_{S} {(S-Mu)' *Q* (S_Mu)/2}, such that A <= 0

        :param ctx: context object
        :param Mu: mean in (B,NumSurfaces, W) size
        :param Sigma2Reciprocal: the reciprocal of variance in (B,NumSurfaces, W) size
        :param A: constraint matrix A <= 0 with A of size(n-1, n)
        :return: SOptimal: the optimal S, surface location in (B,NumSurfaces, W) size
                 MInverse:  (2n-1)*(2n-1) matrix,where n= Numsurfaces

        '''
        Q = Sigma2Reciprocal
        assert Mu.shape == S0.shape
        B,N, W = Mu.size() # N is numSurfaces
        assert alpha > 1 and epsilon >0
        assert N-1, 1 == Lamda.size()
        S = S0
        while True:
            R =  torch.bmm(Q,S-Mu)+torch.bmm(torch.transpose(A,1,2), Lamda)


        pass


    @staticmethod
    def backward(ctx,dL):
        pass

class SeparationPrimalDualIPM(nn.Module):
    def __init__(self):
        pass

    def forward(self, Mu, Sigma2Reciprocal):
        return SeparationPrimalDualIPMFunction.apply(Mu, Sigma2Reciprocal)


