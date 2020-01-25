
import torch
import torch.nn as nn
import sys
sys.path.append(".")
from OCTOptimization import *

class SeparationPrimalDualIPMFunction(torch.autograd.Function):
    @staticmethod
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

        :return: S: the optimal S, surface location in (B, W,N) size
                 MInverse:  (2n-1)*(2n-1) matrix,where n= NumSurfaces, save for backward

        '''
        assert Mu.shape == S0.shape
        B, W, N = Mu.shape  # N is numSurfaces
        assert torch.all(alpha > 1) and epsilon > 0
        Mu = Mu.unsqueeze(dim=-1)  # in size: B,W,N,1
        S0 = S0.unsqueeze(dim=-1)  # in size:B,W,N,1
        Lambda = Lambda.unsqueeze(dim=-1)  # in size: B,W, N-1,1
        alpha = alpha.unsqueeze(dim=-1).unsqueeze(dim=-1)  # in size: B,W,1,1
        S = S0

        m = N - 1
        beta1 = 0.55
        beta2 = 0.055

        AS = torch.matmul(A, S0)  # in size: B,W, N-1, 1
        DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1

        while True:
            S0 = S
            Lambda0 = Lambda

            # update t
            t = -alpha * m / torch.matmul(AS.transpose(-1, -2), Lambda)  # t is in (B,W, 1,1) size

            # compute primal dual search direction
            R = SeparationPrimalDualIPMFunction.getResidualMatrix(Q, S0, Mu, A, Lambda, t, AS,
                                                                  DLambda)  # in size: B,W, 2N-1,1
            M = torch.cat(
                (torch.cat((Q, A.transpose(-1, -2)), dim=-1),
                 torch.cat((torch.matmul(DLambda, A), -torch.diag_embed(AS.squeeze(dim=-1))), dim=-1)),
                dim=-2)  # in size: B,W,2N-1,2N-1
            MInv = torch.inverse(M)
            PD = -torch.matmul(MInv, R)  # primal dual improve direction, in size: B,W,2N-1,1
            PD_S = PD[:, :, 0:N, :]  # in size: B,W,N,1
            PD_Lambda = PD[:, :, N:, :]  # in size: B,W,N-1,1

            # linear search to determine step
            # make sure updated Lambda >=0
            negPDLambda = (PD_Lambda < 0).int() * PD_Lambda + (PD_Lambda >= 0).int() * (
                -1) * Lambda  # in size: B,W, N-1,1
            step, _ = (-Lambda / negPDLambda).min(dim=-2, keepdim=True)
            step = 0.99 * step  # in size: B,W,1,1

            # make sure AS<0
            stepExpandN = step.expand_as(PD_S)  # in size: B,W,N,1
            stepExpandN_1 = step.expand(B, W, N - 1, 1)
            S = S0 + stepExpandN * PD_S
            AS = torch.matmul(A, S)  # AS update
            while torch.any(AS > 0):
                step = torch.where(AS > 0, stepExpandN_1 * beta1, stepExpandN_1)
                step, _ = step.min(dim=-2, keepdim=True)  # in size: B,W,1,1
                stepExpandN = step.expand_as(PD_S)  # in size: B,W,N,1
                stepExpandN_1 = step.expand(B, W, N - 1, 1)
                S = S0 + stepExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update

            # make sure norm2 of R reduce
            RNorm = torch.norm(R, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            Lambda = Lambda0 + stepExpandN_1 * PD_Lambda
            DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1
            R2 = SeparationPrimalDualIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t, AS, DLambda)
            R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            while torch.any(R2Norm > (1 - beta2 * step) * RNorm):
                step = torch.where(R2Norm > (1 - beta2 * step) * RNorm, step * beta1, step)
                stepExpandN = step.expand_as(PD_S)  # in size: B,W,N,1
                stepExpandN_1 = step.expand(B, W, N - 1, 1)
                S = S0 + stepExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update
                Lambda = Lambda0 + stepExpandN_1 * PD_Lambda
                DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1
                R2 = SeparationPrimalDualIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t, AS, DLambda)
                R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1

            # print (f"R2Norm = \n{R2Norm}")
            if R2Norm.max() < epsilon:
                break

        # ctx.save_for_backward(Mu, Q, S, MInv) # save_for_backward is just for input and outputs

        if torch.is_grad_enabled():
            ctx.Mu = Mu
            ctx.Q = Q
            ctx.S = S
            ctx.MInv = MInv

        S = S.squeeze(dim=-1)  # in size: B,W,N
        S.requires_grad_(requires_grad=torch.is_grad_enabled())

        return S


    @staticmethod
    def backward(ctx, dL):
        # Mu, Q, S, MInv = ctx.saved_tensors

        dMu = dQ = dA = dS0 = dLambda = dalpha = depsilon = None
        if torch.is_grad_enabled():
            device = dL.device
            Mu = ctx.Mu
            Q = ctx.Q
            S = ctx.S
            MInv = ctx.MInv

            assert dL.dim() == 3
            B, W, N = dL.shape
            assert MInv.shape[2] == 2 * N - 1
            dL = dL.unsqueeze(dim=-1)
            dL_sLambda = torch.cat((dL, torch.zeros(B, W, N - 1, 1, device=device) ), dim=-2)  # size: B,W,2N-1,1
            d_sLambda = -torch.matmul(torch.transpose(MInv, -1, -2), dL_sLambda)  # size: B,W,2N-1,1
            ds = d_sLambda[:, :, 0:N, :]  # in size: B,W,N,1
            if ctx.needs_input_grad[1]:
                dQ = torch.matmul(ds, torch.transpose(S - Mu, -1, -2)) # size: B,W, N,N
            dMu = -torch.matmul(Q, ds) # size: B,W,N,1
            dMu = dMu.squeeze(dim=-1) # size: B,W,N

        return dMu, dQ, dA, dS0, dLambda, dalpha, depsilon



    @staticmethod
    def getResidualMatrix(Q,S,Mu,A, Lambda, t, AS, DLambda):
        B, W, N, _ = Mu.shape  # N is numSurfaces
        Ones = torch.ones(B,W,N-1,1, device=Mu.device)
        R1 = torch.matmul(Q, S - Mu) + torch.matmul(torch.transpose(A, -1, -2), Lambda)  # the upper part of residual matrix R, in size:B,W,N,1
        R2 = torch.matmul(DLambda, AS) - Ones / t.expand_as(Ones)  # the lower part of residual matrixt R, in size: B,W, N-1,1
        R = torch.cat((R1, R2), dim=-2)  # in size: B,W, 2N-1,1
        return R







class SeparationPrimalDualIPM(nn.Module):
    def __init__(self, B,W,N,device=torch.device('cuda:0')):
        '''
        :param B: BatchSize
        :param W: Image width
        :param N: number of surfaces
        '''
        super().__init__()

        # define A, Lambda, alpha, epsilon here which all are non-learning parameter
        A = (torch.eye(N, N, device=device) + torch.diag(torch.ones(N - 1, device=device) * -1, 1))[0:-1] # for s_i - s_{i+1} <= 0 constraint
        A = A.unsqueeze(dim=0).unsqueeze(dim=0)
        self.m_A = A.expand(B, W, N - 1, N)

        self.m_Lambda = torch.rand(B, W, N - 1, device=device)
        self.m_alpha = 10 + torch.rand(B, W, device=device)  # enlarge factor for t
        self.m_epsilon = 0.001

    def forward(self, Mu, sigma2):
        '''

        :param Mu: mean of size (B,N,W), where N is the number of surfaces
        :param Sigma2: variance of size (B,N,W), where N is number of surfaces
        :return:
                S: the optimized surface location in (B,N,W) size
        '''
        # compute S0 here
        with torch.no_grad():
            batchLIS = getBatchLIS_gpu(Mu)
            S0 = gauranteeSurfaceOrder(Mu, batchLIS)
            if torch.all(Mu.eq(S0)):
                return Mu

        # get Q from sigma2
        # Q: diagonal Reciprocal of variance in (B,W,N,N) size
        Q = getQFromVariance(sigma2) # in B,W,N,N

        # switch H and W axis ordder
        MuIPM = Mu.transpose(dim0=-1,dim1=-2)
        S0IPM = S0.transpose(dim0=-1,dim1=-2)

        S = SeparationPrimalDualIPMFunction.apply(MuIPM, Q, self.m_A, S0IPM, self.m_Lambda, self.m_alpha, self.m_epsilon) # in size: B,W,N

        # switch H and W axis order back
        S = S.transpose(dim0=-1,dim1=-2) # in size:B,N,W
        return S




