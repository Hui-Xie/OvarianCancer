# General constrained quadratic IPM opimtization module



import torch
import torch.nn as nn
import sys
sys.path.append(".")
# from OCTOptimization import *

class ConstrainedIPMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b, A, d, S0, Lambda0, alpha, epsilon):
        '''
        the forward of constrained quadratic optimization with Primal-Dual Interior Point Method.

        S* = arg min_{S} {0.5*s^T*H*s+b^T*s+c}, such that AS <= d

        :param ctx: autograd context object
        :param H: size(B,W,N,N)
        :param b: size(B,W,N,1)
        :param A: size(B,W,M,N) constraint matrix AS <= d
        :param d: size(B,W,M,1)
        :param S0: the initial feasible solution, in (B,W, N) size
        :param Lambda0: the initial Lagragian dual variable, in (B,W,M) size
        :param alpha: IPM iteration t enlarge variable, in (B,W) size, alpha >1
        :param epsilon: float scalar, error tolerance

        :return: S: the optimal solution S in (B, W,N) size

        :save for backward propagation:
                 Lambda: the optimal dual variable in (B,W,N) size
                 MInverse:  (M+N)*(M+N) matrix

        '''
        assert Mu.shape == S0.shape
        B, W, N = Mu.shape  # N is numSurfaces
        assert torch.all(alpha > 1) and epsilon > 0
        Mu = Mu.unsqueeze(dim=-1)  # in size: B,W,N,1
        S0 = S0.unsqueeze(dim=-1)  # in size:B,W,N,1
        Lambda0 = Lambda0.unsqueeze(dim=-1)  # in size: B,W, N-1,1
        alpha = alpha.unsqueeze(dim=-1).unsqueeze(dim=-1)  # in size: B,W,1,1

        m = N - 1
        beta1 = 0.5  # step shrink coefficient
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
            t = -alpha * m / torch.matmul(AS.transpose(-1, -2), Lambda)  # t is in (B,W, 1,1) size

            # compute primal dual search direction according S0 and Lambda0 value
            R = ConstrainedIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t, AS, DLambda)  # in size: B,W, 2N-1,1
            M = torch.cat(
                (torch.cat((Q, A.transpose(-1, -2)), dim=-1),
                 torch.cat((torch.matmul(DLambda, A), -torch.diag_embed(AS.squeeze(dim=-1))), dim=-1)),
                dim=-2)  # in size: B,W,2N-1,2N-1

            try:
                MInv = torch.inverse(M)
            except RuntimeError as err:
                if "singular U" in str(err):
                    # use pseudo-inverse to handle singular square matrix, but pinverse costs 10-20 times of time of inverse.
                    MInv = torch.pinverse(M)
                else:
                    raise RuntimeError(err)

            PD = -torch.matmul(MInv, R)  # primal dual improve direction, in size: B,W,2N-1,1
            PD_S = PD[:, :, 0:N, :]  # in size: B,W,N,1
            PD_Lambda = PD[:, :, N:, :]  # in size: B,W,N-1,1

            # linear search to determine a feasible step which will guarantee constraints
            # make sure updated Lambda >=0
            negPDLambda = (PD_Lambda < 0).int() * PD_Lambda + (PD_Lambda >= 0).int() * (
                -1) * Lambda  # in size: B,W, N-1,1
            step = -(Lambda / negPDLambda)   # in torch.tensor 0/0 = nan, and -0/(-2) = -0, and -(0/(-2)) = 0
            step = torch.where(step != step, torch.ones_like(step), step)  # replace nan as 1
            step, _ = step.min(dim=-2, keepdim=True)
            step = 0.99 * step  # in size: B,W,1,1
            # assert (step == step).all().item()  # detect nan because of nan!=nan

            # make sure AS<0
            stepExpandN = step.expand_as(PD_S)  # in size: B,W,N,1
            stepExpandN_1 = step.expand_as(PD_Lambda) # in size: B,W,N-1,1
            S = S0 + stepExpandN * PD_S
            AS = torch.matmul(A, S)  # AS update
            while torch.any(AS > 0):
                step = torch.where(AS > 0, stepExpandN_1 * beta1, stepExpandN_1)
                step, _ = step.min(dim=-2, keepdim=True)  # in size: B,W,1,1
                stepExpandN = step.expand_as(PD_S)  # in size: B,W,N,1
                stepExpandN_1 = step.expand_as(PD_Lambda)  # in size: B,W,N-1,1
                S = S0 + stepExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update

            # make sure norm2 of R reduce
            RNorm = torch.norm(R, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            Lambda = Lambda0 + stepExpandN_1 * PD_Lambda
            DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1
            R2 = ConstrainedIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t, AS, DLambda)
            R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            while torch.any(R2Norm > (1 - beta2 * step) * RNorm):
                step = torch.where(R2Norm > (1 - beta2 * step) * RNorm, step * beta1, step)
                stepExpandN = step.expand_as(PD_S)  # in size: B,W,N,1
                stepExpandN_1 = step.expand_as(PD_Lambda) # in size: B,W,N-1,1
                S = S0 + stepExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update
                Lambda = Lambda0 + stepExpandN_1 * PD_Lambda
                DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,N-1,N-1
                R2 = ConstrainedIPMFunction.getResidualMatrix(Q, S, Mu, A, Lambda, t, AS, DLambda)
                R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1

            nIPMIterations +=1
            if R2Norm.max() < epsilon or nIPMIterations >7: # IPM generally iterates 6-7 iterations.
                break

        # print(f"Primal-dual IPM nIterations = {nIPMIterations}")
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
                SDiffMu = S - Mu
                dQ = 0.5(torch.matmul(ds, torch.transpose(SDiffMu, -1, -2))+ torch.matmul(SDiffMu, torch.transpose(ds, -1, -2))) # size: B,W, N,N
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




# below is an application of IPM optimization function.
class SoftConstrainedIPMModule(nn.Module):
    def __init__(self ):
        '''
        :param B: BatchSize
        :param W: Image width
        :param N: number of surfaces, length of optimization vector.

        totally this ConstrainedIPM class has B*W optimizations parallel.
        '''
        super().__init__()


    def forward(self, c_lambda, Mu, sigma2, R):
        '''

        :param c_lambda > 0: the balance weight of unary terms;
        :param Mu: mean of size (B,N,W), where N is the number of surfaces
        :param Sigma2: variance of size (B,N,W), where N is number of surfaces
        :param R: learned rift of adjacent surfaces, in size (B,N,W)
        :return:
                S: the optimized surface location in (B,N,W) size
        '''

        B,N,W = Mu.shape
        device = Mu.device

        # compute initial feasible point of the optimization variable
        with torch.no_grad():
            S0, _ = torch.sort(Mu,dim=1)
            S0 = S0.transpose(dim0=-1, dim1=-2) # in size: B,W,N

        # construct H and b matrix
        H = torch.zeros((B,W,N,N), device=device)
        b = torch.zeros((B,W,N,1), device=device)

        Mu = Mu.transpose(dim0=-1,dim1=-2) # in size:B,W,N
        sigma2 = sigma2.transpose(dim0=-1,dim1=-2) # in size:B,W,N
        R = R.transpose(dim0=-1,dim1=-2) # in size:B,W,N
        for i in range(N):
            H[:,:,i,i] +=c_lambda/sigma2[:,:,i] +2.0
            b[:,:,i,0] +=-c_lambda*Mu[:,:,i]/sigma2[:,:,i]-2*R[:,:,i]
            if i > 0:
                H[:,:,i-1,i-1] += 2.0
                H[:, :, i , i - 1] += -4.0
                b[:, :, i-1, 0] += 2*R[:,:,i]

        # according to different application, define A, Lambda, alpha, epsilon here which are non-learning parameter
        A = (torch.eye(N, N, device=device) + torch.diag(torch.ones(N - 1, device=device) * -1, 1))[0:-1]  # for s_i - s_{i+1} <= 0 constraint
        A = A.unsqueeze(dim=0).unsqueeze(dim=0)
        A = A.expand(B, W, N - 1, N)
        d = torch.zeros((B,W,N-1, 1),device=device)

        Lambda0 = torch.rand(B, W, N - 1, device=device)
        alpha = 10 + torch.rand(B, W, device=device)  # enlarge factor for t
        epsilon = 0.01  # 0.001 is too small.

        S = ConstrainedIPMFunction.apply(H, b, A, d, S0, Lambda0, alpha, epsilon) # in size: B,W,N

        # switch H and W axis order back
        S = S.transpose(dim0=-1,dim1=-2) # in size:B,N,W
        return S




