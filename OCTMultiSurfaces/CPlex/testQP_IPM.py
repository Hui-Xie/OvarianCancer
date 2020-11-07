
import sys
import cplex
import torch
import os
import time

print("Comparison between IPM and Cplex")
sys.path.append("..")
sys.path.append("../..")
from network.QuadraticIPMOpt import ConstrainedIPMFunction

device = torch.device('cuda:0')

B = 155
W = 512
N = 12
M = N-1  # number of constraint
print(f"B={B},W={W},N={N},M={M}")

A = (torch.eye(N, N, device=device) + torch.diag(torch.ones(M, device=device) * -1, 1))[0:-1]  # for s_i - s_{i+1} <= 0 constraint
A = A.unsqueeze(dim=0).unsqueeze(dim=0)
A = A.expand(B, W, M, N)
d = torch.zeros((B,W,M, 1),device=device) # relax

# a bigger lambda may increase IPM step(alpha)
Lambda0 = 20*torch.rand(B, W, M, 1, device=device) # size: (B,W,M,1)
beta3 = 10 + torch.rand(B, W, 1, 1, device=device)  # enlarge factor for t, size:(B,W,1,1)
epsilon = 0.0001  # 0.001 is too small.

MaxIterations = 250

H = torch.rand((N,N),device=device)
H = torch.mm(H,H.t()) + torch.eye(N, N, device=device)  # HxH' to make PSD
H = H.unsqueeze(dim=0).unsqueeze(dim=0)
H = H.expand(B, W, N, N)

b = torch.rand((N,1),device=device)
b = b.unsqueeze(dim=0).unsqueeze(dim=0)
b = b.expand(B,W, N, 1)

S0 = list(range(N))
S0 = torch.tensor(S0, device=device, dtype=torch.float)
S0 = S0.view(N,1)
S0 = S0.unsqueeze(dim=0).unsqueeze(dim=0)
S0 = S0.expand(B,W, N, 1)




# IPM solution
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
startTime = time.time()
S = ConstrainedIPMFunction.apply(H, b, A, d, S0, Lambda0, beta3, epsilon, MaxIterations) # in size: B,W,N,1
print(f"IPM running time: {time.time()- startTime} seconds")
print(f"IPM solution S[0,0]: \n", S[0,0])
print(f"========End of IPM======")

objName = []
for i in range(N):
    objName.append(f"s{i}")
sense = 'L'*M

startTime = time.time()

for bi in range(B):
    for wi in range(W):
        Qmat = H[bi,wi].tolist()
        bvec = b[bi, wi].view(N).tolist()
        Amat = A[bi, wi].tolist()
        dvec = d[bi, wi].view(M).tolist()

        p = cplex.Cplex()
        p.set_log_stream(None)
        p.set_error_stream(None)
        p.set_warning_stream(None)
        p.set_results_stream(None)
        p.objective.set_sense(p.objective.sense.minimize)

        Qindex = list(range(N))
        Q = []
        for i in range(N):
            Q.append([Qindex, Qmat[i]])
        objCoef = bvec
        ub = [cplex.infinity, ]*N
        lb = [-cplex.infinity,]*N
        p.variables.add(obj=objCoef, ub=ub, lb=lb, names=objName)
        p.objective.set_quadratic(Q)  # this line must be place after p.variable.add()

        # constraint matrix in row mode
        constraintIndex= list(range(N))
        constraintMatrix = []
        for i in range(M):
            constraintMatrix.append([constraintIndex, Amat[i]])
        rhs = dvec
        p.linear_constraints.add(lin_expr=constraintMatrix, senses=sense,  rhs=rhs)


        p.solve()

        if 0 == bi and 0==wi:
            # solution.get_status() returns an integer code
            print("Solution status = ", p.solution.get_status(), ":", end=' ')
            # the following line prints the corresponding string
            print(p.solution.status[p.solution.get_status()])
            print("Solution value  = ", p.solution.get_objective_value())

            print(f"Cplex running time: {time.time()- startTime} seconds")

            numcols = p.variables.get_num()
            for j in range(numcols):
                print("Column ", j, ":  ", end=' ')
                print("Value = %10f " % p.solution.get_values(j), end=' ')
                print("Reduced Cost = %10f" % p.solution.get_reduced_costs(j))
        # print(f"===============End of bi={bi}, wi={wi}=============\n")

print(f"CPlex running time: {time.time()- startTime} seconds")
print(f"========End of Cplex======")