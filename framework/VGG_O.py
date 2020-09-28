# refer VGG modle

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
        self.m_features = gerateVGGFeatureLayers(VGGCfgs['A'], inputC, batch_norm=True)

        self.m_dropout = nn.Dropout(p=hps.dropoutRate, inplace=True)

    def forward(self,x):
        x = self.m_features(x)

        # Traditionally: global average on H,and W dimension, each feature plane
        # global pooling measures the concentration of feature values in each channel
        x = x.mean(dim=(2, 3), keepdim=True)  # size: B,outputChannels,1,1

        x = self.m_dropout(x)
        return x

