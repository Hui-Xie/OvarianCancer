
# add config Parameter Dict file for old network

netPath = "/home/hxie1/data/OCT_Beijing/numpy/10FoldCVForMultiSurfaceNet/netParameters/OCTUnet/expUnet_20191228_CV0"

import torch
import os

configParameterDict={}
configParameterDict["validationLoss"] = 0.4109
configParameterDict["epoch"] = 2992

torch.save(configParameterDict, os.path.join(netPath, "ConfigParameters.pt"))

print(f"added file in {netPath}")
