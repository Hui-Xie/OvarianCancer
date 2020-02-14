
import torch

configDir = "/home/hxie1/data/IVUS/polarNumpy/netParameters/IVUSUnet/expUnetIVUS_20200214"
configFile = configDir + "/ConfigParameters.pt"

configDict = torch.load(configFile)

print(f"Information in {configFile}\n")
for key, value in configDict.items():
    print(f"\t{key}:{value}")

