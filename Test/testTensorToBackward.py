import torch

dtype = torch.float32
device1 = torch.device('cuda:0')
device2 = torch.device('cuda:1')
x = torch.tensor([2.0], dtype=dtype, device=device1, requires_grad=True)
y = x.to(device=device2)
y = y.pow(3)


y.backward()
print(f"x.gradient= {x.grad}")
print("======END========")
