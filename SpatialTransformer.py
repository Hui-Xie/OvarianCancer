# ResNeXt Block
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialTransformer(nn.Module):
    def __init__(self, inChannels, midChannels, inputx, inputy):
        super().__init__()
        x,y = inputx, inputy
        self.m_localization = nn.ModuleList()
        self.m_localization.append(nn.Conv2d(inChannels, midChannels, kernel_size=1, stride=1, padding=0, bias=False))
        self.m_localization.append(nn.BatchNorm2d(midChannels))
        self.m_localization.append(nn.ReLU(inplace=True))
        while (x>50 and y >50):
            self.m_localization.append(nn.Conv2d(midChannels, midChannels, kernel_size=3, stride=2, padding=0, dilation=3, bias=False))
            x = math.floor((x-3*(3-1)-1)/2+1)
            y = math.floor((y-3*(3-1)-1)/2+1)
            self.m_localization.append(nn.MaxPool2d(kernel_size=3, stride=2))
            x = math.floor((x - 1 * (3 - 1) - 1) / 2 + 1)
            y = math.floor((y - 1 * (3 - 1) - 1) / 2 + 1)
            self.m_localization.append(nn.BatchNorm2d(midChannels))
            self.m_localization.append(nn.ReLU(inplace=True))

        self.m_regression = nn.Linear(midChannels*x*y, 6)

        # Initialize the weights/bias with identity transformation
        self.m_regression.weight.data.zero_()
        self.m_regression.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = x
        for layer in self.m_localization:
            xs = layer(xs)
        xs = torch.reshape(xs, (xs.shape[0], xs.numel() // xs.shape[0]))
        xs = self.m_regression(xs)
        theta = xs.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x
