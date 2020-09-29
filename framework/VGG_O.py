# refer VGG modle

import torch
import torch.nn as nn

VGGCfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def gerateVGGFeatureLayers(cfg, in_channels=3, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG_O(nn.Module):
    '''
    This model refer to VGG's feature layers.
    '''
    def __init__(self, inputC, outputC, hps=None):
        super().__init__()
        self.hps = hps
        self.m_features = gerateVGGFeatureLayers(VGGCfgs['E'], inputC, batch_norm=True)

        self.m_dropout = nn.Dropout(p=hps.dropoutRate, inplace=True)

    def forward(self,x):
        B, _, _, _ = x.shape
        x = self.m_features(x)

        if self.hps.useGlobalMean:
            # Traditionally: global average on H,and W dimension, each feature plane
            # global pooling measures the concentration of feature values in each channel
            x = x.mean(dim=(2, 3), keepdim=True)  # size: B,outputChannels,1,1
        else:
            # Non-traditionally: Use IQR+std on H,and W dimension, each feature plane
            # IQR+std measures the spread of feature values in each channel
            xStd = torch.std(x, dim=(2, 3), keepdim=True)  # size: B,hps.outputChannels, 1,1
            # below 2 row is to solve the NaN probelm
            xMean = torch.mean(x, dim=(2, 3), keepdim=True)  # size: B,hps.outputChannels, 1,1
            xStd = torch.where(xStd > 1e+3, xMean, xStd)

            xFeatureFlat = x.view(B, self.hps.outputChannels, -1)
            xSorted, _ = torch.sort(xFeatureFlat, dim=-1)
            B, C, N = xSorted.shape
            Q1 = N // 4
            Q3 = N * 3 // 4
            # InterQuartile Range
            IQR = (xSorted[:, :, Q3] - xSorted[:, :, Q1]).unsqueeze(dim=-1).unsqueeze(dim=-1)
            x = xStd + IQR

        x = self.m_dropout(x)
        return x

