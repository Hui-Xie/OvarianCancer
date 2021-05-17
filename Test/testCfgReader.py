
import sys
sys.path.append("../framework")
from ConfigReader import ConfigReader
import os

import torch


cfgFilePath = "/home/hxie1/projects/DeepLearningSeg/OCTMultiSurfaces/SoftSepar3Unet/testConfig_IVUS/expIVUS_20210517_SurfaceSubnetQ64_100percent_B_skm2.yaml"

hps = ConfigReader(cfgFilePath)

a=0
a = a+1
print("I love this game")
hps.plane = 3

print(f"hps.plane = {hps.plane}")

goodBscans= hps.goodBscans
#print(goodBscans['2639'])

with open("/localscratch/Users/hxie1/temp/output_test.txt", "w") as file:
    hps.printTo(file)

if "weightL1Loss" in hps.__dict__:
    weightL1 = hps.weightL1Loss
else:
    weightL1 = 1.0
print(f"weightL1 = {weightL1}")

# get GPU information
N = torch.cuda.device_count()
print(f"GPU number={N}")

print(os.system("pwd"))


print("=================end of Config Reader===========")
