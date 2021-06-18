
import torch  # need to eval network name.
import yaml
import os

class ConfigReader(object):
    def __init__(self, yamlFilePath):
        baseName = os.path.basename(yamlFilePath)
        baseName = baseName[0: baseName.find(".yaml")]
        self.experimentName = baseName
        self.yamlFilePath = yamlFilePath
        self.read()

    def read(self):
        with open(self.yamlFilePath) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        self.__dict__.update(cfg)

        self.device = eval(self.device) if "device" in self.__dict__ else None

        if "IVUS" not in self.dataDir:
            self.rotation = False

        if "" != self.loadNetPath:
            self.netPath = self.loadNetPath
        else:
            self.netPath = os.path.join(cfg["netPath"], self.network,self.experimentName)
        if not os.path.exists(self.netPath):
            os.makedirs(self.netPath)  # recursive dir creation

        if ("logDir" not in self.__dict__) or ("" == self.logDir):
            self.logDir = os.path.join(self.dataDir, "log", self.network, self.experimentName)
        else:
            self.logDir = os.path.join(self.logDir, self.network, self.experimentName)
        if not os.path.exists(self.logDir):
            os.makedirs(self.logDir)  # recursive dir creation

        if len(self.outputDir) < 3:
            self.outputDir = os.path.join(self.logDir,"testResult")
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)  # recursive dir creation

        #Add a log path for some log memo output
        self.logMemoPath = os.path.join(self.outputDir, f"logMemo.txt")

        self.imagesOutputDir = os.path.join(self.outputDir, "images")
        self.xmlOutputDir = os.path.join(self.outputDir, "xml")
        self.validationOutputDir = os.path.join(self.outputDir, "validation")
        self.testOutputDir = os.path.join(self.outputDir, "test")
        if not os.path.exists(self.imagesOutputDir):
            os.makedirs(self.imagesOutputDir)  # recursive dir creation
        if not os.path.exists(self.xmlOutputDir):
            os.makedirs(self.xmlOutputDir)  # recursive dir creation
        if not os.path.exists(self.validationOutputDir):
            os.makedirs(self.validationOutputDir)  # recursive dir creation
        if not os.path.exists(self.testOutputDir):
            os.makedirs(self.testOutputDir)  # recursive dir creation

    def printTo(self, fileHandle):
        fileHandle.write(f"\n=============== Start of Config  ===========\n")
        [fileHandle.write(f"{key}:{value}\n") for key, value in vars(self).items()]
        fileHandle.write(f"=============== End of Config  ===========\n\n")