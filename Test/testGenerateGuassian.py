import sys
sys.path.append("..")
from OCTMultiSurfaces.network.OCTAugmentation import gaussianizeLabels
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device =torch.device("cuda:0")
rawLabels = torch.tensor([[80, 90, 70],[100, 120, 100]], device=device)
sigma = 20
H = 512

outputDir = "/home/hxie1/data/temp"
gaussianTensor =  gaussianizeLabels(rawLabels, sigma, H)

N,H,W = gaussianTensor.shape


f = plt.figure(frameon=False)
DPI = f.dpi
f.set_size_inches(H/ float(DPI), 100/ float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

plt.plot(range(0, H), gaussianTensor[1, :, 0].cpu().numpy(), linewidth=1)
#plt.axis('off')

sum = torch.sum(gaussianTensor,dim=(1))

print (f"sum = {sum}")

print (f"a gaussian sample: {gaussianTensor[1, :, 0]}")

plt.savefig(os.path.join(outputDir, "generateGuassian.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()

# test KLDiv loss
GT = gaussianTensor[1, :, 0] # mean at 100's gaussian
GTMaxProb,  GTMaxLoc= torch.max(GT,dim=0)

P = torch.rand(GT.shape).to(device)
P[GTMaxLoc:GTMaxLoc+5] = 10

klDivLoss = nn.KLDivLoss(reduction='batchmean').to(device)  # the input given is expected to contain log-probabilities
loss_surfaceKLDiv = klDivLoss(nn.LogSoftmax(dim=0)(P), GT)

print(f"loss_surfaceKLDiv = {loss_surfaceKLDiv}")






print ("==============end of program ====")

