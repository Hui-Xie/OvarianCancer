import torch

# conclusion: Dropout is on the channel, and it can use after conv.

x = torch.rand(5,10,1,1)
print(f"x=\n{x.view(5,10)}")
dropout = torch.nn.Dropout(p=0.5)
z = dropout(x)
print(f"after dropout z=\n{z.view(5,10)}")

print("============")

#