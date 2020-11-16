
import torch

configDir = "/home/hxie1/data/OCT_Duke/numpy_slices/netParameters/RiftSubnet/expDuke_20200902A_RiftSubnet"
configFile = configDir + "/ConfigParameters.pt"

configDict = torch.load(configFile)

print(f"Information in {configFile}\n")
for key, value in configDict.items():
    print(f"\t{key}:{value}")

