import torch
from torch import autograd

class Foo(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        s = x[0:10]
        # s =x.clone()[0:10] # if we  use this line to replace its above line, error disappears
        ctx.save_for_backward(x,s)
        return s

    @staticmethod
    def backward(ctx, gx):
        x,s = ctx.saved_tensors
        return s.clone()

inp = torch.rand(100, requires_grad=True)

print(f"torchversion: {torch.__version__}")

with torch.no_grad():
    Foo.apply(inp).sum()
print("Ok")