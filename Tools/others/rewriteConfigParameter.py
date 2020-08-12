
import torch

configDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/netParameters/SurfacesUnet/expTongren_9Surfaces_SoftConst_20200808_CV0_8Grad_LearnPair_Pretrain_LR1/realtime"
configFile = configDir + "/ConfigParameters.pt"

configDict={}
configDict["validationLoss"] =17.1
configDict["epoch"]=361
configDict["errorMean"] = 2.203

torch.save(configDict, configFile)

