import torch
import torch.nn as nn
import torch.nn.functional as F

class SegVModel (nn.Module):
    def __init__(self):
        super(SegVModel, self).__init__()

