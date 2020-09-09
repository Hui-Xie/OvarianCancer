# test mobileNet v3

import torch
import torch.optim as optim
import sys
sys.path.append("..")
from framework.MobileNetV3 import MobileNetV3

net = MobileNetV3(3, 3)

gt =torch.tensor([1.0,2.0,3.0])

optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=0)

input = torch.rand(4,3,10,10)
for i in range(300):
    x = input
    y = net(x)
    loss = torch.sum((y-gt.view_as(y))**2)
    optimizer.zero_grad()
    loss.backward(gradient=torch.ones(loss.shape))
    optimizer.step()
    print(f"i={i}; loss={loss.item()}; y = {y.view(1,3)}")

print(f"========End==========")


