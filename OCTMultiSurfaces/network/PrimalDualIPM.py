
import torch
import torch.nn as nn

class SeparationPrimalDualIPMFunction(torch.autograd.Function):

    @staticmethod
    def backward(ctx,dL):
        pass


def forward(ctx, Mu, Q, A, S0, Lambda, alpha, epsilon):
    '''
    S* = arg min_{S} {(S-Mu)' *Q* (S-Mu)/2}, such that A <= 0

    :param ctx: context object
    :param Mu: predicted mean,  in (B,W, NumSurfaces) size, in below N=NumSurfaces
    :param Q: Sigma2Reciprocal: Q, the diagonal reciprocal of variance in (B,W,N,N) size
    :param A: constraint matrix A <= 0 with A of size(n-1, n), in (B, W, N-1,N) size
    :param S0: in (B,W, N) size
    :param Lambda: dual variable, in (B,W,N-1) size
    :param alpha: IPM iteration t enlarge variable, in (B,W) size, alpha >1
    :param epsilon: float scalar, error tolerance

    :return: S: the optimal S, surface location in (B,NumSurfaces, W) size
             MInverse:  (2n-1)*(2n-1) matrix,where n= NumSurfaces, save for backward

    '''
    assert Mu.shape == S0.shape
    B,W,N = Mu.size() # N is numSurfaces
    assert torch.all(alpha > 1) and epsilon >0
    Mu = Mu.unsqueeze(dim=-1)  # in size: B,W,N,1
    S0 = S0.unsqueeze(dim=-1)  # in size:B,W,N,1
    Lambda = Lambda.unsqueeze(dim=-1) # in size: B,W, N-1,1
    alpha = alpha.unsqueeze(dim=-1).unsqueeze(dim=-1) # in size: B,W,1,1
    S = S0
    Ones = torch.ones(B,W,N-1,1)
    m = N-1

    while True:
        AS = torch.matmul(A,S)
        DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1

        # update t
        t = -alpha*m / torch.matmul(AS.transpose(-1,-2), Lambda)    # t is in (B,W, 1,1) size

        # compute primal dual search direction
        R1 = torch.matmul(Q,S-Mu)+torch.matmul(torch.transpose(A,-1,-2), Lambda) # the upper part of residual matrix R, in size:B,W,N,1
        R2 = torch.matmul(DLambda, AS)- Ones/t.expand_as(Ones)  # the lower part of residual matrixt R, in size: B,W, N-1,1
        R = torch.cat((R1,R2),dim=-2) # in size: B,W, 2N-1,1
        M = torch.cat(
            (torch.cat((Q, A.transpose(-1,-2)), dim=-1),
            torch.cat((torch.matmul(DLambda, A), -torch.diag_embed(AS.squeeze(dim=-1))), dim=-1)),
            dim=-2)  # in size: B,W,2N-1,2N-1
        MInv = torch.inverse(M)
        PD = -torch.matmul(MInv,R) # primal dual improve direction, in size: B,W,2N-1,1
        PD_S = PD[:,:,0:N,:]
        PD_Lambda = PD[:,:,N:,:]

        # linear search to determine step
        negPDLambda = (PD_Lambda<0).int()*PD_Lambda # in size: B,W, N-1,1









    pass


class SeparationPrimalDualIPM(nn.Module):
    def __init__(self):
        pass

    def forward(self, Mu, Sigma2Reciprocal):
        return SeparationPrimalDualIPMFunction.apply(Mu, Sigma2Reciprocal)


