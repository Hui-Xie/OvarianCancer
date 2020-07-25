
import torch

configDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet_10Surfaces_AllGoodBscans/netParameters/SurfacesUnet/expTongren_10Surfaces_AllGood_SoftConst_20200723_CV0"
configFile = configDir + "/ConfigParameters.pt"

configDict = torch.load(configFile)

print(f"Information in {configFile}\n")
for key, value in configDict.items():
    print(f"\t{key}:{value}")

