# ResNeXt Block
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SpatialTransformer(nn.Module):
    def __init__(self, inChannels, midChannels, inputW, inputH, useSpectralNorm=False, useLeakyReLU=False):
        super().__init__()
        w,h = inputW, inputH
        self.m_localization = nn.ModuleList()
        self.m_localization.append(nn.Conv2d(inChannels, midChannels, kernel_size=1, stride=1, padding=0, bias=True) if not useSpectralNorm
                                   else nn.utils.spectral_norm(nn.Conv2d(inChannels, midChannels, kernel_size=1, stride=1, padding=0, bias=True)))
        self.m_localization.append(nn.BatchNorm2d(midChannels))
        self.m_localization.append(nn.ReLU(inplace=True) if not useLeakyReLU else nn.LeakyReLU(inplace=True))
        while (w>=7 and h >=7):
            self.m_localization.append(nn.Conv2d(midChannels, midChannels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True) if not useSpectralNorm
                                       else nn.utils.spectral_norm(nn.Conv2d(midChannels, midChannels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True)))
            w = math.floor((w-1*(3-1)-1)/1+1)
            h = math.floor((h-1*(3-1)-1)/1+1)
            self.m_localization.append(nn.MaxPool2d(kernel_size=3, stride=2))
            w = math.floor((w - 1 * (3 - 1) - 1) / 2 + 1)
            h = math.floor((h - 1 * (3 - 1) - 1) / 2 + 1)
            self.m_localization.append(nn.BatchNorm2d(midChannels))
            self.m_localization.append(nn.ReLU(inplace=True) if not useLeakyReLU else nn.LeakyReLU(inplace=True))
        self.m_localization.append(nn.LocalResponseNorm(midChannels))  # normalization along channels

        self.m_regression = nn.Linear(midChannels*w*h, 7)  # affine 6 elements + modulation factor
        #if useSpectralNorm:
        #     self.m_regression = nn.utils.spectral_norm(self.m_regression)

        # Initialize the weights/bias with identity transformation
        self.m_regression.weight.data.zero_()
        self.m_regression.bias.data[0:6].copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # for theta with size(2,3)
        self.m_eps = 1e-12
        self.m_u = F.normalize(torch.empty(2, dtype=torch.float, requires_grad=False).normal_(0, 1), dim=0, eps=self.m_eps)
        self.m_v = F.normalize(torch.empty(3, dtype=torch.float, requires_grad=False).normal_(0, 1), dim=0, eps=self.m_eps)

    def forward(self, x):
        xs = x
        for layer in self.m_localization:
            xs = layer(xs)
        xs = torch.reshape(xs, (xs.shape[0], xs.numel() // xs.shape[0]))
        xs = self.m_regression(xs)

        theta = xs[:,0:6].clone()
        m = xs[:,6].clone()      # modulation factor
        m = torch.sigmoid(m)     # convert into range [0,1]

        theta = theta.view(-1, 2, 3)

        theta = self.spectralNormalize(theta)  # Spectral Normalize to reduce image repeating.

        grid = F.affine_grid(theta, x.size())
        xout = F.grid_sample(x, grid, padding_mode="reflection")

        # modulate the output of STN
        batch = x.shape[0]
        for i in range(batch):
            s = xout[i,].clone()
            xout[i,] = s*m[i]
        return xout

    def spectralNormalize(self, theta):
        batch = theta.shape[0]
        u = self.m_u.clone().to(theta.device)
        v = self.m_v.clone().to(theta.device)
        for i in range(batch):
            W = theta[i,].clone()  #Gradients propagating to the cloned tensor will propagate to the original tensor.
            for _ in range(4): #power_iteration
                v = F.normalize(torch.mv(W.t(), u), dim=0, eps=self.m_eps)
                u = F.normalize(torch.mv(W, v), dim=0, eps=self.m_eps)
            maxSingleValue =  torch.dot(u, torch.mv(W, v))
            theta[i,] = W/maxSingleValue
        return theta
