
import torch
import torch.nn as nn
import sys
sys.path.append(".")
from OCTOptimization import *

class MuQIPMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Mu, Q, A, S0, Lambda0, beta3, epsilon):
        '''
        the forward of constrained convex optimization with Primal-Dual Interior Point Method.

        S* = arg min_{S} {(S-Mu)' *Q* (S-Mu)/2}, such that AS <= 0

        :param ctx: autograd context object
        :param Mu: predicted mean,  in (B,W, NumSurfaces, 1) size, in below N=NumSurfaces
        :param Q: Sigma2Reciprocal: Q, the diagonal reciprocal of variance in (B,W,N,N) size
        :param A: constraint matrix AS <= 0 with A of size(n-1, n), in (B, W, N-1,N) size
        :param S0: the initial feasibible solution, in (B,W, N, 1) size
        :param Lambda0: the initial Lagragian dual variable, in (B,W,N-1, 1) size
        :param beta3: IPM iteration t enlarge variable, in (B,W, 1,1) size, beta3 >1
        :param epsilon: float scalar, error tolerance

        :return: S: the optimal solution S, surface location in (B, W,N) size
                 MInverse:  (2n-1)*(2n-1) matrix,where n= NumSurfaces, save for backward

        '''
        assert Mu.shape == S0.shape
        B, W, N,_ = Mu.shape  # N is numSurfaces
        assert torch.all(beta3 > 1) and epsilon > 0
        
        M = N - 1
        beta1 = 0.5  # alpha shrink coefficient
        beta2 = 0.055

        S = S0
        Lambda = Lambda0

        nIPMIterations = 0
        while True:
            # preserve the previous iteration as S0  and Lambda0
            # while S and Lambda indicate current S and Lambda
            S0 = S.clone()
            Lambda0 = Lambda.clone()

            AS = torch.matmul(A, S)  # in size: B,W, N-1, 1
            DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1

            # update t
            t = -beta3 * M / torch.matmul(AS.transpose(-1, -2), Lambda)  # t is in (B,W, 1,1) size

            # compute primal dual search direction according S0 and Lambda0 value
            R = MuQIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t)  # in size: B,W, 2N-1,1
            J = torch.cat(
                (torch.cat((Q, A.transpose(-1, -2)), dim=-1),
                 torch.cat((torch.matmul(DLambda, A), -torch.diag_embed(AS.squeeze(dim=-1))), dim=-1)),
                dim=-2)  # in size: B,W,2N-1,2N-1

            try:
                J_Inv = torch.inverse(J)
            except RuntimeError as err:
                if "singular U" in str(err):
                    # use pseudo-inverse to handle singular square matrix, but pinverse costs 10-20 times of time of inverse.
                    J_Inv = torch.pinverse(J)
                else:
                    raise RuntimeError(err)

            PD = -torch.matmul(J_Inv, R)  # primal dual improve direction, in size: B,W,2N-1,1
            PD_S = PD[:, :, 0:N, :]  # in size: B,W,N,1
            PD_Lambda = PD[:, :, N:, :]  # in size: B,W,N-1,1

            # linear search to determine a feasible alpha which will guarantee constraints
            # make sure updated Lambda >=0
            negPDLambda = (PD_Lambda < 0).int() * PD_Lambda + (PD_Lambda >= 0).int() * (-1) * Lambda  # in size: B,W, N-1,1
            alpha = -(Lambda / negPDLambda)   # in torch.tensor 0/0 = nan, and -0/(-2) = -0, and -(0/(-2)) = 0
            alpha = torch.where(alpha != alpha, torch.ones_like(alpha), alpha)  # replace nan as 1
            alpha, _ = alpha.min(dim=-2, keepdim=True)
            alpha = 0.99 * alpha  # in size: B,W,1,1
            # assert (alpha == alpha).all().item()  # detect nan because of nan!=nan

            # make sure AS<0
            alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
            alphaExpandN_1 = alpha.expand_as(PD_Lambda) # in size: B,W,N-1,1
            S = S0 + alphaExpandN * PD_S
            AS = torch.matmul(A, S)  # AS update
            while torch.any(AS > 0):
                alpha = torch.where(AS > 0, alphaExpandN_1 * beta1, alphaExpandN_1)
                alpha, _ = alpha.min(dim=-2, keepdim=True)  # in size: B,W,1,1
                alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
                alphaExpandN_1 = alpha.expand_as(PD_Lambda)  # in size: B,W,N-1,1
                S = S0 + alphaExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update

            # make sure norm2 of R reduce
            RNorm = torch.norm(R, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            Lambda = Lambda0 + alphaExpandN_1 * PD_Lambda
            R2 = MuQIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t)
            R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            while torch.any(R2Norm > (1 - beta2 * alpha) * RNorm):
                alpha = torch.where(R2Norm > (1 - beta2 * alpha) * RNorm, alpha * beta1, alpha)
                alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
                alphaExpandN_1 = alpha.expand_as(PD_Lambda) # in size: B,W,N-1,1
                S = S0 + alphaExpandN * PD_S
                Lambda = Lambda0 + alphaExpandN_1 * PD_Lambda
                R2 = MuQIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t)
                R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1

            nIPMIterations +=1
            if R2Norm.max() < epsilon or nIPMIterations >7: # IPM generally iterates 6-7 iterations.
                break

        # print(f"Primal-dual IPM nIterations = {nIPMIterations}")
        # ctx.save_for_backward(Mu, Q, S, J_Inv) # save_for_backward is just for input and outputs

        # these stash variables need free in the backward
        ctx.Mu = Mu
        ctx.Q = Q
        ctx.S = S
        ctx.J_Inv = J_Inv

        return S


    @staticmethod
    def backward(ctx, dL):
        # Mu, Q, S, J_Inv = ctx.saved_tensors

        dMu = dQ = dA = dS0 = dLambda = dalpha = depsilon = None

        device = dL.device
        Mu = ctx.Mu
        Q = ctx.Q
        S = ctx.S
        J_Inv = ctx.J_Inv

        assert dL.dim() == 4
        B, W, N, One = dL.shape
        assert J_Inv.shape[2] == 2 * N - 1
        dL_sLambda = torch.cat((dL, torch.zeros(B, W, N - 1, 1, device=device) ), dim=-2)  # size: B,W,2N-1,1
        d_sLambda = -torch.matmul(torch.transpose(J_Inv, -1, -2), dL_sLambda)  # size: B,W,2N-1,1
        ds = d_sLambda[:, :, 0:N, :]  # in size: B,W,N,1
        if ctx.needs_input_grad[1]:
            dQ = torch.matmul(torch.diag_embed(ds.squeeze(dim=-1)), (S-Mu).transpose(dim0=-1, dim1=-2).expand(B, W, N, N))
            # amos gradient formula:
            # SDiffMu = S - Mu
            # dQ = 0.5*(torch.matmul(ds, torch.transpose(SDiffMu, -1, -2))+ torch.matmul(SDiffMu, torch.transpose(ds, -1, -2))) # size: B,W, N,N
        dMu = -torch.matmul(Q.transpose(dim0=-1, dim1=-2), ds) # size: B,W,N,1

        # free stash variables.
        ctx.Mu = None
        ctx.Q = None
        ctx.S = None
        ctx.J_Inv = None

        return dMu, dQ, dA, dS0, dLambda, dalpha, depsilon



    @staticmethod
    def getResidualMatrix(Q,S,Mu,A, Lambda, t):
        B, W, N, _ = Mu.shape  # N is numSurfaces
        Ones = torch.ones(B,W,N-1,1, device=Mu.device)
        AS = torch.matmul(A, S)  # in size: B,W, N-1, 1
        DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1
        R1 = torch.matmul(Q, S - Mu) + torch.matmul(torch.transpose(A, -1, -2), Lambda)  # the upper part of residual matrix R, in size:B,W,N,1
        R2 = torch.matmul(DLambda, AS) - Ones / t.expand_as(Ones)  # the lower part of residual matrixt R, in size: B,W, N-1,1
        R = torch.cat((R1, R2), dim=-2)  # in size: B,W, 2N-1,1
        return R







class HardSeparationIPMModule(nn.Module):
    def __init__(self):
        '''
        application layer
        '''
        super().__init__()

        

    def forward(self, Mu, sigma2):
        '''
         s^* = argmin \sum_{i\in[0,N)}\{\lambda \frac{(s_{i}-\mu_{i})^2}{2\sigma_{i}^2}\}
         
        :param Mu: mean of size (B,N,W), where N is the number of surfaces
        :param Sigma2: variance of size (B,N,W), where N is number of surfaces
        :return:
                S: the optimized surface location in (B,N,W) size
        '''
        
        B,N,W = Mu.shape
        device = Mu.device

        # Optimization will not optimize sigma
        # sigma2 = sigma2.detach()

        # compute S0 here
        with torch.no_grad():
            batchLIS = getBatchLIS_gpu(Mu)
            S0 = guaranteeSurfaceOrder(Mu, batchLIS)
            if torch.all(Mu.eq(S0)):
                return Mu

        # get Q from sigma2
        # Q: diagonal Reciprocal of variance in (B,W,N,N) size
        Q = getQFromVariance(sigma2) # in B,W,N,N

        # switch H and W axis ordder
        MuIPM = Mu.transpose(dim0=-1,dim1=-2).unsqueeze(dim=-1)
        S0IPM = S0.transpose(dim0=-1,dim1=-2).unsqueeze(dim=-1)

        # define A, Lambda, beta3, epsilon here which are non-learning parameter
        M=N-1
        A = (torch.eye(N, N, device=device) + torch.diag(torch.ones(M, device=device) * -1, 1))[
            0:-1]  # for s_i - s_{i+1} <= 0 constraint
        A = A.unsqueeze(dim=0).unsqueeze(dim=0)
        A = A.expand(B, W, M, N)

        Lambda0 = torch.rand(B, W, M, 1, device=device)
        beta3 = 10 + torch.rand(B, W, 1, 1,device=device)  # enlarge factor for t
        epsilon = 0.01  # 0.001 is too small.

        S = MuQIPMFunction.apply(MuIPM, Q, A, S0IPM, Lambda0, beta3, epsilon) # in size: B,W,N

        # switch H and W axis order back
        S = S.squeeze(dim=-1)  # in size: B,W,N
        S = S.transpose(dim0=-1,dim1=-2) # in size:B,N,W
        return S




