# General constrained quadratic IPM opimtization module
import torch
import torch.nn as nn

class ConstrainedIPMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, b, A, d, S0, Lambda0, beta3, epsilon, nMaxIteration):
        '''
        the forward of constrained quadratic optimization using Primal-Dual Interior Point Method.

        S^* = arg min_{S} {0.5*s^T*H*s+b^T*s+c}, such that AS <= d

        :param ctx: autograd context object
        :param H: size(B,W,N,N)
        :param b: size(B,W,N,1)
        :param A: size(B,W,M,N) constraint matrix AS <= d
        :param d: size(B,W,M,1)
        :param S0: the initial feasible solution, in (B,W, N, 1) size
        :param Lambda0: the initial Lagragian dual variable, in (B,W,M,1) size
                        A bigger lambda may increase IPM step(alpha), reduces iterative number.
        :param beta3: IPM iteration t enlarge variable, in (B,W, 1, 1) size, beta3 >1
        :param epsilon: float scalar, error tolerance

        :return: S: the optimal solution S in (B, W,N, 1) size

        :save for backward propagation in ctx:
                 Lambda: the optimal dual variable in (B,W,N, 1) size
                 J_Inv:  (M+N)*(M+N) matrix of reverse of J

        '''
        B, W, N,One = S0.shape  # N is the length of optimization variable
        B1,W1,M,N1 = A.shape
        assert B==B1 and W==W1 and N==N1 and One==1
        assert torch.all(beta3 > 1) and epsilon > 0
        beta1 = 0.5  # alpha shrink coefficient
        beta2 = 0.055

        device = H.device

        S = S0
        Lambda = Lambda0

        nIPMIterations = 0
        while True:
            # preserve the previous iteration as S0  and Lambda0
            # while S and Lambda indicate current S and Lambda
            S0 = S.clone()
            Lambda0 = Lambda.clone()

            # update t
            AS = torch.matmul(A, S)  # in size: B,W, M, 1
            t = -beta3 * M / torch.matmul((AS-d).transpose(-1, -2), Lambda)  # t is in (B,W, 1,1) size

            # compute primal dual search direction according S and Lambda value
            R = ConstrainedIPMFunction.getResidualMatrix(H, b, A, d, S, Lambda, t)  # in size: B,W, N+M,1
            # the Jacobian matrix
            J = torch.cat(
                (torch.cat((H, A.transpose(-1, -2)), dim=-1),
                 torch.cat((torch.matmul(-torch.diag_embed(Lambda.squeeze(dim=-1)), A),
                            -torch.diag_embed((AS-d).squeeze(dim=-1))), dim=-1)),
                 dim=-2)  # in size: B,W,N+M,N+M

            try:
                J_Inv = torch.inverse(J)  # in size: B,W,N+M,N+M
            except RuntimeError as err:
                if "singular U" in str(err):
                    # use pseudo-inverse to handle singular square matrix, but pinverse costs 10-20 times of time of inverse.
                    J_Inv = torch.pinverse(J)
                else:
                    raise RuntimeError(err)

            # Inverse using the SVD can create nan problems(not converge)
            # when the singular values are not unique or very close each other.
            if torch.isnan(J_Inv.sum()):
                turbulence = (1e-4*J.mean().abs()*torch.eye(N+M, device=device)).unsqueeze(dim=0).unsqueeze(dim=0).expand(B,W,N+M, N+M) # size: BxWx(N+M)x(N+M)
                J += turbulence
                J_Inv = torch.inverse(J)
                if torch.isnan(J_Inv.sum()):
                    print(f"**Error**: in IPM forward, the inverse of Jacobian can not converge even adding small turbulence.")
                    ctx.svdError = True
                    return S  # return a trivial solution


            PD = -torch.matmul(J_Inv, R)  # primal dual improve direction, in size: B,W,N+M,1
            PD_S = PD[:, :, 0:N, :]  # in size: B,W,N,1
            PD_Lambda = PD[:, :, N:, :]  # in size: B,W,M,1

            # linear search to determine a feasible alpha which will guarantee constraints and make sure updated Lambda >=0
            negPDLambda = (PD_Lambda < 0).int() * PD_Lambda + (PD_Lambda >= 0).int() * (-1) * Lambda  # in size: B,W, M,1
            alpha = -(Lambda / negPDLambda)   # in torch.tensor 0/0 = nan, and -0/(-2) = -0, and -(0/(-2)) = 0
            alpha = torch.where(alpha != alpha, torch.ones_like(alpha), alpha)  # replace nan as 1
            alpha, _ = alpha.min(dim=-2, keepdim=True)
            alpha = 0.99 * alpha  # in size: B,W,1,1
            # assert (alpha == alpha).all().item()  # detect nan because of nan!=nan

            # make sure AS<0
            alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
            alphaExpandM = alpha.expand_as(PD_Lambda) # in size: B,W,M,1
            S = S0 + alphaExpandN * PD_S
            AS = torch.matmul(A, S)  # AS update
            while torch.any(AS > d):
                alpha = torch.where(AS > d, alphaExpandM * beta1, alphaExpandM)
                alpha, _ = alpha.min(dim=-2, keepdim=True)  # in size: B,W,1,1
                alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
                alphaExpandM = alpha.expand_as(PD_Lambda)  # in size: B,W,M,1
                S = S0 + alphaExpandN * PD_S
                AS = torch.matmul(A, S)  # AS update

            # make sure the norm2 of R reduce
            RNorm = torch.norm(R, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            Lambda = Lambda0 + alphaExpandM * PD_Lambda
            R2 = ConstrainedIPMFunction.getResidualMatrix(H, b, A, d, S, Lambda, t)
            R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1
            while torch.any(R2Norm > (1 - beta2 * alpha) * RNorm):
                alpha = torch.where(R2Norm > (1 - beta2 * alpha) * RNorm, alpha * beta1, alpha)
                alphaExpandN = alpha.expand_as(PD_S)  # in size: B,W,N,1
                alphaExpandM = alpha.expand_as(PD_Lambda) # in size: B,W,M,1
                S = S0 + alphaExpandN * PD_S
                Lambda = Lambda0 + alphaExpandM * PD_Lambda
                R2 = ConstrainedIPMFunction.getResidualMatrix(H, b, A, d, S, Lambda, t)
                R2Norm = torch.norm(R2, p=2, dim=-2, keepdim=True)  # size: B,W,1,1

            nIPMIterations +=1
            if R2Norm.max() < epsilon or nIPMIterations >nMaxIteration: # IPM generally iterates 6-7 iterations.
                #debug
                #print(f"IPM forward iterations = {nIPMIterations}, and R2Norm.max = {R2Norm.max()}")
                break

        # print(f"Primal-dual IPM nIterations = {nIPMIterations}")
        # here torch.is_grad_enabled()= false
        ctx.S = S
        ctx.J_Inv = J_Inv
        ctx.Lambda = Lambda
        ctx.svdError = False
        # for verify IPM forward use
        # ctx.R = R2
        # ctx.t = t
        # S.requires_grad_(requires_grad=torch.is_grad_enabled())
        return S


    @staticmethod
    def backward(ctx, dL):
        dH = db = dA = dd = dS0 = dLambda0 = dbeta3 = depsilon = dMaxIterations=None

        if ctx.svdError:
            ctx.svdError = None
            return dH, db, dA, dd, dS0, dLambda0, dbeta3, depsilon, dMaxIterations

        # in backward, all torch.is_grad_enabled() is false by autograd mechanism
        device = dL.device
        S = ctx.S
        J_Inv = ctx.J_Inv
        Lambda = ctx.Lambda

        assert dL.dim() == 4
        B, W, N,One = dL.shape
        B1,W1,M,One1 = Lambda.shape
        assert B==B1 and W==W1 and One==One1==1
        assert J_Inv.shape[2] == M+N
        dL_sLambda = torch.cat((dL, torch.zeros(B, W, M, 1, device=device) ), dim=-2)  # size: B,W,N+M,1
        d_sLambda = -torch.matmul(torch.transpose(J_Inv, -1, -2), dL_sLambda)  # size: B,W,N+M,1
        ds = d_sLambda[:, :, 0:N, :]  # in size: B,W,N,1
        dlambda = d_sLambda[:, :, N:, :]  # in size: B,W,M,1
        if ctx.needs_input_grad[0]:
            # amos solution:
            # dH = 0.5*(torch.matmul(S, ds.transpose(dim0=-1,dim1=-2))+torch.matmul(ds, S.transpose(dim0=-1,dim1=-2)))  # size: B,W,N,N
            dH = torch.matmul(torch.diag_embed(ds.squeeze(dim=-1)),S.transpose(dim0=-1,dim1=-2).expand(B,W,N,N))
            dH = torch.where(dH != dH, torch.zeros_like(dH), dH)  # replace nan as 0
        if ctx.needs_input_grad[1]:
            db = ds  # size: B,W, N,1
            db = torch.where(db != db, torch.zeros_like(db), db)  # replace nan as 0
        if ctx.needs_input_grad[2]:
            DLambda = torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M
            Ddlambda = torch.diag_embed(dlambda.squeeze(dim=-1))
            dA = torch.matmul(DLambda,
                              (ds.transpose(dim0=-1,dim1=-2).expand(B,W,M,N)- torch.matmul(Ddlambda, S.transpose(dim0=-1,dim1=-2).expand(B,W,M,N))))
            dA = torch.where(dA != dA, torch.zeros_like(dA), dA)  # replace nan as 0
            # amos solution
            # DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M
            # dA = torch.matmul(Lambda, ds.transpose(dim0=-1,dim1=-2))+torch.matmul(DLambda, torch.matmul(dlambda, S.transpose(dim0=-1,dim1=-2)))

        if ctx.needs_input_grad[3]:
            DLambda = torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M
            dd = torch.matmul(DLambda,dlambda)
            dd = torch.where(dd != dd, torch.zeros_like(dd), dd)  # replace nan as 0
        # free self added ctx fields
        ctx.S = None
        ctx.J_Inv = None
        ctx.Lambda = None
        ctx.svdError = None
        # for verify IPM forward use
        # ctx.R = None
        # ctx.t = None

        return dH, db, dA, dd, dS0, dLambda0, dbeta3, depsilon, dMaxIterations



    @staticmethod
    def getResidualMatrix(H, b, A, d, S, Lambda, t):
        B, W, M, N = A.shape  # N is the number of optimization variable
        Ones = torch.ones(B,W,M,1, device=A.device)
        AS = torch.matmul(A, S)  # in size: B,W, M, 1
        DLambda = -torch.diag_embed(Lambda.squeeze(dim=-1))  # negative diagonal Lambda in size:B,W,M,M
        R1 = torch.matmul(H, S) +b+ torch.matmul(torch.transpose(A, -1, -2), Lambda)  # the upper part of residual matrix R, in size:B,W,N,1
        R2 = torch.matmul(DLambda, AS-d) - Ones / t.expand_as(Ones)  # the lower part of residual matrixt R, in size: B,W, M,1
        R = torch.cat((R1, R2), dim=-2)  # in size: B,W, N+M,1
        return R




# below is an application of IPM optimization function.
class SoftSeparationIPMModule(nn.Module):
    def __init__(self ):
        '''
        An application layer for soft constrained quadratic IPM optimization.
        This ConstrainedIPM class has total B*W parallel optimizations.
        '''
        super().__init__()


    def forward(self, Mu, sigma2, R=None, fixedPairWeight=False, learningPairWeight=None):
        '''
        s^* = argmin \sum_{i\in[0,N)}\{ \frac{(s_{i}-\mu_{i})^2}{2\sigma_{i}^2} + \lambda (s_{i}-s_{i-1} -r_{i})^2 \}

        :param Mu: mean of size (B,N,W), where N is the number of surfaces
        :param Sigma2: variance of size (B,N,W), where N is number of surfaces
        :param R: learned rift of adjacent surfaces, in size (B,N-1,W)
                  if R==None, means it is pure unary terms cost function.
        :param fixedPairWeight: True or False
        :param learningPairWeight:  in size Bx(N-1)xW
        :return:
                S: the optimized surface location in (B,N,W) size
        '''

        B,N,W = Mu.shape
        device = Mu.device
        if fixedPairWeight:
            # if use fixedPairWeight, learningPairWeight should be None
            assert (learningPairWeight is None)

        # Optimization will not optimize sigma and R.
        # sigma2 = sigma2.detach()
        # R = RR.clone().detach()

        # compute initial feasible point of the optimization variable
        # Here detach is unnecessary, as operation in autograd_function does not compute gradient
        with torch.no_grad():
            # ReLU to guarantee layer order not to cross each other
            S0 = Mu.clone()
            for i in range(1, N):
                S0[:, i, :] = torch.where(S0[:, i, :] < S0[:, i - 1, :], S0[:, i - 1, :], S0[:, i, :])
            S0 = S0.transpose(dim0=-1, dim1=-2) # in size: B,W,N
            S0 = S0.unsqueeze(dim=-1)  #in size: B,W,N,1
        S0_detach = S0.clone().detach()
        # construct H and b matrix

        # c is the unary weight in the cost function
        # c = (4.1*sigma2.max()).detach().item()
        if (R is not None):
            if (learningPairWeight is not None):
                c = torch.ones((B,W),device=device)
            else:
                c,_ = torch.max(sigma2, dim=1, keepdim=False) # lambda, size: BxW
                c = (20.0*c).detach() # size: BxW  # 4c is 50% weight; while 16c is 80% weight, 20c is 83% weight;

        H = torch.zeros((B,W,N,N), device=device)
        b = torch.zeros((B,W,N,1), device=device)

        Mu = Mu.transpose(dim0=-1,dim1=-2) # in size:B,W,N
        sigma2 = sigma2.transpose(dim0=-1,dim1=-2) # in size:B,W,N
        sigma2 = sigma2 +1e-8  # avoid sigma2 ==0

        if learningPairWeight is not None:
            learningPairWeight = learningPairWeight.transpose(dim0=-1,dim1=-2) # in size:B,W,N-1, avoid H singular

        if R is not None:  # soft separation constraint
            R = R.transpose(dim0=-1,dim1=-2) # in size:B,W,N-1
            # with (N-1) rifts for N surfaces
            for i in range(N):
                if 0==i:
                    H[:,:,i,i] +=c[:,:]/sigma2[:,:,i]  # H_0
                    b[:, :, i, 0] += -c[:, :] * Mu[:, :, i] / sigma2[:, :, i]  #b_0
                else:
                    if fixedPairWeight:
                        pairwiceC = sigma2[:, :, i]/(sigma2[:, :, i] + sigma2[:, :, i-1])
                    elif (learningPairWeight is not None):
                        pairwiceC = learningPairWeight[:,:,i-1]
                    else:
                        pairwiceC = 1  # for pairwise weight = 1 case
                    H[:, :, i, i] += c[:, :] / sigma2[:, :, i] + 2.0*pairwiceC
                    H[:, :, i - 1, i - 1] += 2.0 * pairwiceC
                    H[:, :, i, i - 1] += -4.0 * pairwiceC

                    b[:, :, i, 0] += -c[:, :] * Mu[:, :, i] / sigma2[:, :, i] - 2 * R[:, :, i-1]*pairwiceC
                    b[:, :, i-1, 0] += 2*R[:,:,i-1]*pairwiceC
        else:  # for pure hard separation constraint
            for i in range(N):
                H[:,:,i,i] +=1.0/sigma2[:,:,i]
                b[:, :, i, 0] += -Mu[:, :, i] / sigma2[:, :, i]

        # according to different application, define A, Lambda, beta3, epsilon here which are non-learning parameter
        M = N-1  # number of constraints
        A = (torch.eye(N, N, device=device) + torch.diag(torch.ones(M, device=device) * -1, 1))[0:-1]  # for s_i - s_{i+1} <= 0 constraint
        A = A.unsqueeze(dim=0).unsqueeze(dim=0)
        A = A.expand(B, W, M, N)
        d = torch.zeros((B,W,M, 1),device=device) # relax

        # a bigger lambda may increase IPM step(alpha)
        Lambda0 = 20*torch.rand(B, W, M, 1, device=device) # size: (B,W,M,1)
        beta3 = 10 + torch.rand(B, W, 1, 1, device=device)  # enlarge factor for t, size:(B,W,1,1)
        epsilon = 0.01  # 0.001 is too small.

        MaxIterations = 50

        S = ConstrainedIPMFunction.apply(H, b, A, d, S0_detach, Lambda0, beta3, epsilon, MaxIterations) # in size: B,W,N,1

        # switch H and W axis order back
        S = S.squeeze(dim=-1)
        S = S.transpose(dim0=-1,dim1=-2) # in size:B,N,W
        return S




