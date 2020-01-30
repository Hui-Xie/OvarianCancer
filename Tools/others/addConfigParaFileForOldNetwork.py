
# add config Parameter Dict file for old network

netPath = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/netParameters/OCTUnet/expUnet_20191228_CV7"

import torch
import os

configParameterDict={}
configParameterDict["validationLoss"] = 0.4763
configParameterDict["epoch"] = 1949

torch.save(configParameterDict, os.path.join(netPath, "ConfigParameters.pt"))

print(f"added file in {netPath}")
