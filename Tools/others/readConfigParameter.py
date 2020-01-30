
import torch

configFile = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/netParameters/OCTUnet/expUnet_20200117_CV1_DP/ConfigParameters.pt"

configDict = torch.load(configFile)

print(configDict)
