# General constrained quadratic IPM opimtization module



import torch
import torch.nn as nn
import sys
sys.path.append(".")
# from OCTOptimization import *

class ConstrainedIPMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b, A, d, S0, Lambda0, beta3, epsilon):
        '''
        the forward of constrained quadratic optimization with Primal-Dual Interior Point Method.

        S* = arg min_{S} {0.5*s^T*H*s+b^T*s+c}, such that AS <= d

        :param ctx: autograd context object
        :param H: size(B,W,N,N)
        :param b: size(B,W,N,1)
        :param A: size(B,W,M,N) constraint matrix AS <= d
        :param d: size(B,W,M,1)
        :param S0: the initial feasible solution, in (B,W, N, 1) size
        :param Lambda0: the initial Lagragian dual variable, in (B,W,M,1) size
        :param beta3: IPM iteration t enlarge variable, in (B,W, 1, 1) size, beta3 >1
        :param epsilon: float scalar, error tolerance

        :return: S: the optimal solution S in (B, W,N, 1) size

        :save for backward propagation:
                 Lambda: the optimal dual variable in (B,W,N, 1) size
                 J_Inverse:  (M+N)*(M+N) matrix

        '''
        B, W, N,One = S0.shape  # N is the length of optimizatino variable
        B1,W1,M,N1 = A.shape
        assert B==B1 and W==W1 and N==N1 and One==1
        assert torch.all(beta3 > 1) and epsilon > 0
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

            AS = torch.matmul(A, S)  # in size: B,W, M, 1
            DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M

            # update t
            t = -beta3 * M / torch.matmul((AS-d).transpose(-1, -2), Lambda)  # t is in (B,W, 1,1) size

            # compute primal dual search direction according S0 and Lambda0 value
            R = ConstrainedIPMFunction.getResidualMatrix(H, b, A, d, S, Lambda, t, AS, DLambda)  # in size: B,W, N+M,1
            J = torch.cat(
                (torch.cat((H, A.transpose(-1, -2)), dim=-1),
                 torch.cat((torch.matmul(DLambda, A), -torch.diag_embed((AS-d).squeeze(dim=-1))), dim=-1)),
                dim=-2)  # in size: B,W,N+M,N+M

            try:
                J_Inv = torch.inverse(J)  # in size: B,W,N+M,N+M
            except RuntimeError as err:
                if "singular U" in str(err):
                    # use pseudo-inverse to handle singular square matrix, but pinverse costs 10-20 times of time of inverse.
                    J_Inv = torch.pinverse(J)
                else:
                    raise RuntimeError(err)

            PD = -torch.matmul(J_Inv, R)  # primal dual improve direction, in size: B,W,N+M,1
            PD_S = PD[:, :, 0:N, :]  # in size: B,W,N,1
            PD_Lambda = PD[:, :, N:, :]  # in size: B,W,M,1

            # linear search to determine a feasible alpha which will guarantee constraints
            # make sure updated Lambda >=0
            negPDLambda = (PD_Lambda < 0).int() * PD_Lambda + (PD_Lambda >= 0).int() * (-1) * Lambda  # in size: B,W, M,1
            alpha = -(Lambda / negPDLambda)   # in torch.tensor 0/0 = nan, and -0/(-2) = -0, and -(0/(-2)) = 0
            alpha = torch.where(alpha != alpha, torch.ones_like(alpha), alpha)  # replace nan as 1
            alpha, _ = alpha.min(dim=-2, keepdim=True)
            alpha = 0.99 * alpha  # in size: B,W,1,1
            # assert (alpha == alpha).all().item()  # detect nan because of nan!=nan

            # make sure AS<0
            alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
            alphaExpandN_1 = alpha.expand_as(PD_Lambda) # in size: B,W,M,1
            S = S0 + alphaExpandN * PD_S
            AS = torch.matmul(A, S)  # AS update
            while torch.any(AS > d):
                alpha = torch.where(AS > d, alphaExpandN_1 * beta1, alphaExpandN_1)
                alpha, _ = alpha.min(dim=-2, keepdim=True)  # in size: B,W,1,1
                alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
                alphaExpandN_1 = alpha.expand_as(PD_Lambda)  # in size: B,W,M,1
                S = S0 + alphaExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update

            # make sure norm2 of R reduce
            RNorm = torch.norm(R, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            Lambda = Lambda0 + alphaExpandN_1 * PD_Lambda
            DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M
            R2 = ConstrainedIPMFunction.getResidualMatrix(H, b, A, d, S, Lambda, t, AS, DLambda)
            R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            while torch.any(R2Norm > (1 - beta2 * alpha) * RNorm):
                alpha = torch.where(R2Norm > (1 - beta2 * alpha) * RNorm, alpha * beta1, alpha)
                alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
                alphaExpandN_1 = alpha.expand_as(PD_Lambda) # in size: B,W,M,1
                S = S0 + alphaExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update
                Lambda = Lambda0 + alphaExpandN_1 * PD_Lambda
                DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M
                R2 = ConstrainedIPMFunction.getResidualMatrix(H, b, A, d, S, Lambda, t, AS, DLambda)
                R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1

            nIPMIterations +=1
            if R2Norm.max() < epsilon or nIPMIterations >7: # IPM generally iterates 6-7 iterations.
                break

        # print(f"Primal-dual IPM nIterations = {nIPMIterations}")
        # ctx.save_for_backward(Mu, Q, S, MInv) # save_for_backward is just for input and outputs
        if torch.is_grad_enabled():
            ctx.S = S
            ctx.J_Inv = J_Inv
            ctx.Lambda = Lambda

        S = S.squeeze(dim=-1)  # in size: B,W,N
        S.requires_grad_(requires_grad=torch.is_grad_enabled())

        return S


    @staticmethod
    def backward(ctx, dL):
        dH = db = dA = dd = dS0 = dLambda = dbeta3 = depsilon = None
        if torch.is_grad_enabled():
            device = dL.device
            S = ctx.S
            J_Inv = ctx.J_Inv
            Lambda = ctx.Lambda

            assert dL.dim() == 3
            B, W, N = dL.shape
            B1,W1,M,One = Lambda.shape
            assert B==B1 and W==W1 and One==1
            assert J_Inv.shape[2] == M+N
            dL = dL.unsqueeze(dim=-1)
            dL_sLambda = torch.cat((dL, torch.zeros(B, W, M, 1, device=device) ), dim=-2)  # size: B,W,N+M,1
            d_sLambda = -torch.matmul(torch.transpose(J_Inv, -1, -2), dL_sLambda)  # size: B,W,N+M,1
            ds = d_sLambda[:, :, 0:N, :]  # in size: B,W,N,1
            dlambda = d_sLambda[:, :, N:, :]  # in size: B,W,M,1
            if ctx.needs_input_grad[0]:
                dH = 0.5*(torch.matmul(S, ds.transpose(dim0=-1,dim1=-2))+torch.matmul(ds, S.transpose(dim0=-1,dim1=-2)))  # size: B,W,N,N
            if ctx.needs_input_grad[1]:
                db = ds  # size: B,W, N,1
            if ctx.needs_input_grad[2]:
                DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M
                dA = torch.matmul(Lambda, ds.transpose(dim0=-1,dim1=-2))+torch.matmul(DLambda, torch.matmul(dlambda, S.transpose(dim0=-1,dim1=-2)))
            if ctx.needs_input_grad[3]:
                DLambda = torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M
                dd = torch.matmul(DLambda,dlambda)
                
        return dH, db, dA, dd, dS0, dLambda, dbeta3, depsilon



    @staticmethod
    def getResidualMatrix(H, b, A, d, S, Lambda, t, AS, DLambda):
        B, W, M, N = A.shape  # N is numuber of optimization variable
        Ones = torch.ones(B,W,M,1, device=A.device)
        R1 = torch.matmul(H, S) +b+ torch.matmul(torch.transpose(A, -1, -2), Lambda)  # the upper part of residual matrix R, in size:B,W,N,1
        R2 = torch.matmul(DLambda, AS-d) - Ones / t.expand_as(Ones)  # the lower part of residual matrixt R, in size: B,W, M,1
        R = torch.cat((R1, R2), dim=-2)  # in size: B,W, N+M,1
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
            S0 = S0.unsqueeze(dim=-1)  #in size: B,W,N,1

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

        # according to different application, define A, Lambda, beta3, epsilon here which are non-learning parameter
        M = N-1  # number of constraints
        A = (torch.eye(N, N, device=device) + torch.diag(torch.ones(M, device=device) * -1, 1))[0:-1]  # for s_i - s_{i+1} <= 0 constraint
        A = A.unsqueeze(dim=0).unsqueeze(dim=0)

        A = A.expand(B, W, M, N)
        d = torch.zeros((B,W,M, 1),device=device)

        Lambda0 = torch.rand(B, W, M, 1, device=device) # size: (B,W,M,1)
        beta3 = 10 + torch.rand(B, W, 1, 1, device=device)  # enlarge factor for t, size:(B,W,1,1)
        epsilon = 0.01  # 0.001 is too small.

        S = ConstrainedIPMFunction.apply(H, b, A, d, S0, Lambda0, beta3, epsilon) # in size: B,W,N

        # switch H and W axis order back
        S = S.transpose(dim0=-1,dim1=-2) # in size:B,N,W
        return S




