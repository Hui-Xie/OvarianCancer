
import torch

configFile = "/home/hxie1/data/OCT_JHU/numpy/netParameters/OCTUnetSurfaceLayerJHU/expUnetJHU_IPM_SurfaceLayer_20200206/ConfigParameters.pt"

configDict = torch.load(configFile)

print(f"Information in {configFile}\n")
for key, value in configDict.items():
    print(f"\t{key}:{value}")

