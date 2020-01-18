
import torch
import torch.nn as nn

class SeparationPrimalDualIPMFunction(torch.autograd.Function):

    @staticmethod
    def backward(ctx,dL):
        pass

    @staticmethod
    def getResidualMatrix(Q,S,Mu,A, Lambda, t, AS, DLambda):
        B, W, N, _ = Mu.shape  # N is numSurfaces
        Ones = torch.ones(B,W,N-1,1)
        R1 = torch.matmul(Q, S - Mu) + torch.matmul(torch.transpose(A, -1, -2), Lambda)  # the upper part of residual matrix R, in size:B,W,N,1
        R2 = torch.matmul(DLambda, AS) - Ones / t.expand_as(Ones)  # the lower part of residual matrixt R, in size: B,W, N-1,1
        R = torch.cat((R1, R2), dim=-2)  # in size: B,W, 2N-1,1
        return R

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
        B,W,N = Mu.size() # N is numSurfaces
        assert torch.all(alpha > 1) and epsilon >0
        Mu = Mu.unsqueeze(dim=-1)  # in size: B,W,N,1
        S0 = S0.unsqueeze(dim=-1)  # in size:B,W,N,1
        Lambda = Lambda.unsqueeze(dim=-1) # in size: B,W, N-1,1
        alpha = alpha.unsqueeze(dim=-1).unsqueeze(dim=-1) # in size: B,W,1,1
        S = S0

        m = N-1
        beta1 = 0.55
        beta2 = 0.055

        AS = torch.matmul(A, S0)  # in size: B,W, N-1, 1
        DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1

        while True:
            S0 = S
            Lambda0 = Lambda

            # update t
            t = -alpha*m / torch.matmul(AS.transpose(-1,-2), Lambda)    # t is in (B,W, 1,1) size

            # compute primal dual search direction
            R = SeparationPrimalDualIPMFunction.getResidualMatrix(Q,S0,Mu,A, Lambda, t, AS, DLambda)  # in size: B,W, 2N-1,1
            M = torch.cat(
                (torch.cat((Q, A.transpose(-1,-2)), dim=-1),
                torch.cat((torch.matmul(DLambda, A), -torch.diag_embed(AS.squeeze(dim=-1))), dim=-1)),
                dim=-2)  # in size: B,W,2N-1,2N-1
            MInv = torch.inverse(M)
            PD = -torch.matmul(MInv,R) # primal dual improve direction, in size: B,W,2N-1,1
            PD_S = PD[:,:,0:N,:]  # in size: B,W,N,1
            PD_Lambda = PD[:,:,N:,:]  # in size: B,W,N-1,1

            # linear search to determine step
            # make sure updated Lambda >=0
            negPDLambda = (PD_Lambda<0).int()*PD_Lambda + (PD_Lambda>=0).int()*(-1)*Lambda # in size: B,W, N-1,1
            step, _ = (-Lambda/negPDLambda).min(dim=-2, keepdim=True)
            step = 0.99*step # in size: B,W,1,1

            # make sure AS<0
            stepExpandN = step.expand_as(PD_S) # in size: B,W,N,1
            stepExpandN_1 = step.expand(B,W,N-1,1)
            S = S0+stepExpandN*PD_S
            AS = torch.matmul(A, S) # AS update
            while torch.any(AS>0):
                step = torch.where(AS>0,  stepExpandN_1*beta1, stepExpandN_1)
                step, _ = step.min(dim=-2,keepdim=True) # in size: B,W,1,1
                stepExpandN = step.expand_as(PD_S)  # in size: B,W,N,1
                stepExpandN_1 = step.expand(B, W, N - 1, 1)
                S = S0 + stepExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update

            # make sure norm2 of R reduce
            RNorm = torch.norm(R,p=2, dim=-2,keepdim=True) # size: B,W,1,1
            Lambda = Lambda0 + stepExpandN_1*PD_Lambda
            DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1
            R2 = SeparationPrimalDualIPMFunction.getResidualMatrix(Q,S,Mu,A, Lambda, t, AS, DLambda)
            R2Norm = torch.norm(R2,p=2, dim=-2,keepdim=True) # size: B,W,1,1
            while torch.any(R2Norm > (1-beta2*step)*RNorm):
                step = torch.where(R2Norm > (1-beta2*step)*RNorm, step*beta1, step)
                stepExpandN = step.expand_as(PD_S)  # in size: B,W,N,1
                stepExpandN_1 = step.expand(B, W, N - 1, 1)
                S = S0 + stepExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update
                Lambda = Lambda0 + stepExpandN_1 * PD_Lambda
                DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1
                R2 = SeparationPrimalDualIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t, AS, DLambda)
                R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1

            print (f"R2Norm = \n{R2Norm}")
            if R2Norm.max() < epsilon:
                break

        # debug forward, comment it. In the future, it will uncomment
        #ctx.save_for_backward(MInv) # in size: B,W,2N-1,2N-1

        return S # in size: B,W,N,1





class SeparationPrimalDualIPM(nn.Module):
    def __init__(self):
        pass

    def forward(self, Mu, Sigma2Reciprocal):
        return SeparationPrimalDualIPMFunction.apply(Mu, Sigma2Reciprocal)


