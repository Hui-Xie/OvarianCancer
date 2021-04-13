# verify IPM forward and generate related parameters.

import torch
import sys
sys.path.append("../OCTMultiSurfaces/network")
from QuadraticIPMOpt import *

device =torch.device("cuda:0")
dtype = torch.float
B = 1
W = 1
N = 2  # for 2x2 matrix

print(f"Note: Use this program: you need save R and t in IPM forward program")

# construct H and b matrix
'''
case 1:
H = torch.tensor([[1,2],[3,5]], dtype=dtype, device=device)
b = torch.tensor([[-1],[-2.6]], dtype=dtype, device=device)

H = H.unsqueeze(dim=0).unsqueeze(dim=0)  # H size: 1x1x2x2
b = b.unsqueeze(dim=0).unsqueeze(dim=0)  # b size: 1x1x2x1

# according to different application, define A, Lambda, beta3, epsilon
M = N  # number of constraints
A = torch.tensor([[-1,0],[0,-1]], dtype=dtype, device=device)
A = A.unsqueeze(dim=0).unsqueeze(dim=0)  # H size: 1x1x2x2
A = A.expand(B, W, M, N)
d = torch.zeros((B,W,M, 1),device=device)

S0 = torch.matmul(torch.linalg.inv(A),d-2)  # this is a good method to get initial value.
'''


'''
# case 2:
H = torch.tensor([[2,5],[3,11]], dtype=dtype, device=device)
b = torch.tensor([[3],[5]], dtype=dtype, device=device)

H = H.unsqueeze(dim=0).unsqueeze(dim=0)  # H size: 1x1x2x2
b = b.unsqueeze(dim=0).unsqueeze(dim=0)  # b size: 1x1x2x1

# according to different application, define A, Lambda, beta3, epsilon
M = N  # number of constraints
A = torch.tensor([[5,2],[3,4]], dtype=dtype, device=device)
A = A.unsqueeze(dim=0).unsqueeze(dim=0)  # H size: 1x1x2x2
A = A.expand(B, W, M, N)
d = torch.tensor([[2],[3]], dtype=dtype, device=device)
d = d.unsqueeze(dim=0).unsqueeze(dim=0)  # d size: 1x1x2x1

S0 = torch.matmul(torch.linalg.inv(A),d)  # this is a good method to get initial value.

'''

# case 3:
H = torch.tensor([[2,5],[5,11]], dtype=dtype, device=device)
b = torch.tensor([[3],[5]], dtype=dtype, device=device)

H = H.unsqueeze(dim=0).unsqueeze(dim=0)  # H size: 1x1x2x2
b = b.unsqueeze(dim=0).unsqueeze(dim=0)  # b size: 1x1x2x1

# according to different application, define A, Lambda, beta3, epsilon
M = N  # number of constraints
A = torch.tensor([[5,2],[3,4]], dtype=dtype, device=device)
A = A.unsqueeze(dim=0).unsqueeze(dim=0)  # H size: 1x1x2x2
A = A.expand(B, W, M, N)
d = torch.tensor([[2],[3]], dtype=dtype, device=device)
d = d.unsqueeze(dim=0).unsqueeze(dim=0)  # d size: 1x1x2x1

S0 = torch.matmul(torch.linalg.inv(A),d)  # this is a good method to get initial value.


# a bigger lambda may increase IPM step(alpha)
Lambda0 = 20*torch.rand(B, W, M, 1, device=device) # size: (B,W,M,1)
beta3 = 10 + torch.rand(B, W, 1, 1, device=device)  # enlarge factor for t, size:(B,W,1,1)
epsilon = 0.0001  # 0.001 is too small.

MaxIterations = 200

class ContextObject:
    def __init__(self):
        pass

ctx = ContextObject()
S = ConstrainedIPMFunction.forward(ctx, H, b, A, d, S0, Lambda0, beta3, epsilon, MaxIterations) # in size: B,W,N,1
Lambda = ctx.Lambda
R = ctx.R
t = ctx.t

print(f"H = \n{H}")
print(f"b = \n{b}")
print(f"A = \n{A}")
print(f"d = \n{d}")
print(f"S = \n{S}")
print(f"Lambda= \n{Lambda}")
print(f"R = \n{R}")
print(f"t = \n{t}")