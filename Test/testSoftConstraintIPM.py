# test SoftConstrainedIPMModule

import torch
import sys
sys.path.append("../OCTMultiSurfaces/network")
from QuadraticIPMOpt import *
#from OCTPrimalDualIPM import *

from torch.autograd import gradcheck

device =torch.device("cuda:0")
dtype = torch.float
B = 2
W = 3
N = 5
lr = 0.001 # learning rate
nEpic = 100


Mu = 10.0*torch.rand(B,N,W, dtype=dtype, device=device, requires_grad=True)
Mu.retain_grad()
sigma2= 2.0*torch.rand(B,N,W, dtype=dtype, device=device, requires_grad=True)
sigma2.retain_grad()
R = (torch.rand(B,N,W, device=device)+0.5) * torch.cat((Mu[:,0,:].unsqueeze(dim=1), Mu[:,1:,:]- Mu[:,0:-1,:]), dim=1)
R.retain_grad()
#c_lambda = (4*torch.max(sigma2)).clone() # error: one of the variables needed for gradient computation has been modified by an inplace operation:
#c_lambda.detach()
# c_lambda = 10.0

G = torch.tensor([[1,1,1],[3,3,3], [4,4,4], [6,6,6], [8,9,7]], dtype=dtype, device= device)
G = G.unsqueeze(dim=0)
G = torch.cat((G,G),dim=0)  # size:(B,N,W)

# first run IPMModule to get gradient of Input variables.
# test softConstrainedIPM
seperationIPM = SoftSeparationIPMModule()
S = seperationIPM(Mu,sigma2,R)

# test HardSeparationIPMModule
#seperationIPM = HardSeparationIPMModule()
#S = seperationIPM(Mu,sigma2)

loss = torch.pow(G-S,2).sum()

print(f"Before loss.backward: Mu.grad = {Mu.grad}")

# calling the backward() of a Variable only generates the gradients of the leaf nodes
loss.backward(gradient=torch.ones(loss.shape).to(device), retain_graph=True)

print(f"After loss.backward: Mu.grad =\n {Mu.grad}")
print(f"After loss.backward: sigma2.grad =\n {sigma2.grad}")
print(f"After loss.backward: R.grad =\n {R.grad}")

# check whether loss reduces when changing input according gradient.
print(f"epic:{-1}, loss = {loss.item()}")
for i in range(nEpic):
    Mu -= lr* Mu.grad
    sigma2 -= lr*sigma2.grad
    R -= lr * R.grad
    Mu.grad.zero_()
    sigma2.grad.zero_()
    #if R.grad is not None:
    R.grad.zero_()

    # softSeparation
    S = seperationIPM(Mu, sigma2, R)
    # hardSeperation
    #S = seperationIPM(Mu,sigma2)
    loss = torch.pow(G - S, 2).sum()
    print(f"epic:{i}, loss = {loss.item()}")
    loss.backward(gradient=torch.ones(loss.shape).to(device), retain_graph=True)
