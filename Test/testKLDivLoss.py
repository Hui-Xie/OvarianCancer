
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:3")

H = 128
x = torch.rand((H), device=device)
y = torch.rand((H), device=device)
yProb = F.softmax(y, dim=0)

xLogProb = nn.LogSoftmax(dim=0)(x)

klDivLoss = nn.KLDivLoss(reduction='batchmean').to(device)  # the input given is expected to contain log-probabilities
loss_KLDiv = klDivLoss(xLogProb, yProb)

print (f"loss_KLDiv = {loss_KLDiv}")

