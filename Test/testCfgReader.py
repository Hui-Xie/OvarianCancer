
import sys
sys.path.append("../framework")
from ConfigReader import ConfigReader
import os



cfgFilePath = "/home/sheen/Projects/DeepLearningSeg/OCTMultiSurfaces/testConfig/expTongren_10Surfaces_20200509/expTongren_10Surfaces_20200509_CV0.yaml"

hps = ConfigReader(cfgFilePath)

a=0
a = a+1
print("I love this game")
hps.plane = 3

print(f"hps.plane = {hps.plane}")



with open("/home/sheen/temp/output_test.txt", "w") as file:
    hps.printTo(file)


print("=================end of Config Reader===========")
