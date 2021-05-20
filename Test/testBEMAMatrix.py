
import torch
device = torch.device('cuda:0')

W = 100
b = 0.9  # momentum, beta, which denotes the prevous weight of momentum.
m_smoothM = torch.zeros((1, W, W), dtype=torch.float32, device=device,  requires_grad=False)  #smooth matrix

colM = torch.zeros((W, 1), dtype=torch.float32, device=device,  requires_grad=False)  # a column of smooth matrix
colM[0,0] = 1-b
for i in range(1,W//2):
    colM[i,0] = 0.5*pow(b,i)*(1-b)
for i in range(W//2, W):
    colM[i, 0] = 0.5 * pow(b, W-i) * (1 - b)

for i in range(0, W):
    m_smoothM[0, :, i] = colM.roll(i).view(W)

print(m_smoothM)
A= torch.sum(m_smoothM,dim=1)
print(f"A.shpae = {A.shape}")
print(A)