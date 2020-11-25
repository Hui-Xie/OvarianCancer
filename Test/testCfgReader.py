
import sys
sys.path.append("../framework")
from ConfigReader import ConfigReader
import os

import torch


cfgFilePath = "/home/hxie1/Projects/DeepLearningSeg/OCTMultiSurfaces/SoftSepar3Unet/testConfig_Duke/expDuke_20201117A_SurfaceSubnet_NoReLU.yaml"

hps = ConfigReader(cfgFilePath)

a=0
a = a+1
print("I love this game")
hps.plane = 3

print(f"hps.plane = {hps.plane}")

goodBscans= hps.goodBscans
#print(goodBscans['2639'])

with open("/home/hxie1/temp/output_test.txt", "w") as file:
    hps.printTo(file)


# get GPU information
N = torch.cuda.device_count()
print(f"GPU number={N}")

print(os.system("pwd"))


print("=================end of Config Reader===========")
