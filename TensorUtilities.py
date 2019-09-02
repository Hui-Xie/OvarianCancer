
import torch

def zeroMeanNormalize(x, dim=1):
    m = torch.mean(x, dim=dim).clone()
    std = torch.std(x, dim=dim).clone()
    std += 1e-12
    N = x.shape[0]
    xout = torch.empty(x.shape, requires_grad=True).to(device=x.device, dtype=x.dtype)
    for i in range(N):
        xs = x[i,].clone()
        xout[i,] = (xs- m[i])/std[i]
    return xout


