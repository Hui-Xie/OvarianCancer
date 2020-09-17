
import torch
import yaml
import sys
sys.path.append("..")
from utilities.FilesUtilities import getStemName
import os

class ConfigReader(object):
    def __init__(self, yamlFilePath):
        self.experimentName = getStemName(yamlFilePath, removedSuffix=".yaml")
        self.yamlFilePath = yamlFilePath
        self.read()

    def read(self):
        with open(self.yamlFilePath) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.__dict__.update(cfg)

        self.device = eval(self.device)

        if "IVUS" not in self.dataDir:
            self.rotation = False

        if "" != self.loadNetPath:
            self.netPath = self.loadNetPath
        else:
            self.netPath = cfg["netPath"] + "/" + self.network + "/" + self.experimentName

        if ("" == self.logDir) or ("logDir" not in self):
            self.logDir = self.dataDir + "/log/" + self.network + "/" + self.experimentName
        else:
            self.logDir = self.logDir + "/" + self.network + "/" + self.experimentName
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)  # recursive dir creation

        if self.outputDir == "":
            self.outputDir = self.logDir + "/testResult"
            self.imagesOutputDir = self.outputDir +"/images"
            self.xmlOutputDir = self.outputDir + "/xml"
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)  # recursive dir creation
        if not os.path.exists(self.imagesOutputDir):
            os.makedirs(self.imagesOutputDir)  # recursive dir creation
        if not os.path.exists(self.xmlOutputDir):
            os.makedirs(self.xmlOutputDir)  # recursive dir creation

    def printTo(self, fileHandle):
        fileHandle.write(f"\n=============== Start of Config  ===========\n")
        [fileHandle.write(f"{key}:{value}\n") for key, value in vars(self).items()]
        fileHandle.write(f"=============== End of Config  ===========\n\n")