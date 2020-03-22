
import sys
sys.path.append("../framework")
from ConfigReader import ConfigReader
import os



cfgFilePath = "/home/sheen/Projects/DeepLearningSeg/OCTMultiSurfaces/testConfig/expUnetTongren_9Surfaces_20200215/expUnetTongren_20200321_TestCfgReader.yaml"

hps = ConfigReader(cfgFilePath)

a=0
a = a+1
print("I love this game")
hps.plane = 3

print(f"hps.plane = {hps.plane}")



with open("/home/sheen/temp/output_test.txt", "w") as file:
    hps.printTo(file)


print("=================end of Config Reader===========")
