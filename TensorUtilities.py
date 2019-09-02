
import torch

def zeroMeanNormalize(x, dim=1):
    m = torch.mean(x, dim=dim)
    std = torch.std(x, dim=dim)
    std += 1e-12
    N = x.shape[0]
    for i in range(N):
        x[i,] = (x[i,]- m[i])/std[i]
    return x


