# ResNeXt Block
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialTransformer(nn.Module):
    def __init__(self, inChannels, midChannels, inputW, inputH):
        super().__init__()
        w,h = inputW, inputH
        self.m_localization = nn.ModuleList()
        self.m_localization.append(nn.Conv2d(inChannels, midChannels, kernel_size=1, stride=1, padding=0, bias=False))
        self.m_localization.append(nn.BatchNorm2d(midChannels))
        self.m_localization.append(nn.ReLU(inplace=True))
        while (w>25 and h >25):
            self.m_localization.append(nn.Conv2d(midChannels, midChannels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False))
            w = math.floor((w-1*(3-1)-1)/1+1)
            h = math.floor((h-1*(3-1)-1)/1+1)
            self.m_localization.append(nn.MaxPool2d(kernel_size=3, stride=2))
            w = math.floor((w - 1 * (3 - 1) - 1) / 2 + 1)
            h = math.floor((h - 1 * (3 - 1) - 1) / 2 + 1)
            self.m_localization.append(nn.BatchNorm2d(midChannels))
            self.m_localization.append(nn.ReLU(inplace=True))

        self.m_regression = nn.Linear(midChannels*w*h, 6)


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

        # add translation to center
        # shapeTheta = theta.shape
        # shapeX  = x.shape
        # translationCenter= torch.Tensor([[0,0,shapeX[3]//2], [0,0,shapeX[2]//2]]).cuda()
        # for i in range(shapeTheta[0]):
        #      theta[i,] = theta[i,] + translationCenter
        #      rotationSubM = theta[i,:, 0:2]
        #      maxAbs = torch.abs(rotationSubM).max()
        #      rotationSubM /=maxAbs
        #      theta[i,:, 0:2] = rotationSubM


        grid = F.affine_grid(theta, x.size())
        xout = F.grid_sample(x, grid, padding_mode="reflection")
        return xout
