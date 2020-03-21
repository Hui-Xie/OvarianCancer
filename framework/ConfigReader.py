
import torch
import yaml
import sys
sys.path.append("..")
from utilities.FilesUtilities import getStemName
import os

class ConfigReader():
    def __init__(self, yamlFilePath):
        self.yamlFilePath = yamlFilePath
        self.read()


    def read(self):
        with open(self.yamlFilePath) as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)

        self.experimentName = getStemName(self.yamlFilePath, removedSuffix=".yaml")

        self.dataDir = cfg["dataDir"]
        self.K = cfg["K_Folds"]
        self.k = cfg["fold_k"]

        self.groundTruthInteger = cfg["groundTruthInteger"]

        self.sigma = cfg["sigma"]  # for gausssian ground truth
        self.device = eval(cfg["device"])  # convert string to class object.
        self.batchSize = cfg["batchSize"]

        self.network = cfg["network"]  #
        self.inputHight = cfg["inputHight"]  # 192
        self.inputWidth = cfg["inputWidth"]  # 1060  # rawImageWidth +2 *lacingWidth
        self.scaleNumerator = cfg["scaleNumerator"]  # 2
        self.scaleDenominator = cfg["scaleDenominator"]  # 3
        self.inputChannels = cfg["inputChannels"]  # 1
        self.nLayers = cfg["nLayers"]  # 7
        self.numSurfaces = cfg["numSurfaces"]
        self.numStartFilters = cfg["startFilters"]  # the num of filter in first layer of Unet
        self.gradChannels = cfg["gradChannels"]
        self.gradWeight = cfg["gradWeight"]
        self.useLayerCE = cfg['useLayerCE']
        self.useWeightedDivLoss = cfg['useWeightedDivLoss']

        self.slicesPerPatient = cfg["slicesPerPatient"]  # 31
        self.hPixelSize = cfg["hPixelSize"]  # 3.870  # unit: micrometer, in y/height direction

        self.augmentProb = cfg["augmentProb"]
        self.gaussianNoiseStd = cfg["gaussianNoiseStd"]  # gausssian nosie std with mean =0
        # for salt-pepper noise
        self.saltPepperRate = cfg["saltPepperRate"]  # rate = (salt+pepper)/allPixels
        self.saltRate = cfg["saltRate"]  # saltRate = salt/(salt+pepper)
        self.lacingWidth = cfg["lacingWidth"]

        if "IVUS" in self.dataDir:
            self.rotation = cfg["rotation"]
            self.TTA = cfg["TTA"]  # Test-Time Augmentation
            self.TTA_StepDegree = cfg["TTA_StepDegree"]
        else:
            self.rotation = False

        self.netPath = cfg["netPath"] + "/" + self.network + "/" + self.experimentName
        self.loadNetPath = cfg['loadNetPath']
        if "" != self.loadNetPath:
            self.netPath = self.loadNetPath
        self.outputDir = cfg["outputDir"]

        if self.outputDir == "":
            self.outputDir = self.dataDir + "/log/" + self.network + "/" + self.experimentName + "/testResult"
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)  # recursive dir creation

        self.lossFunc0 = cfg["lossFunc0"]  # "nn.KLDivLoss(reduction='batchmean').to(device)"
        self.lossFunc0Epochs = cfg["lossFunc0Epochs"]  # the epoch number of using lossFunc0
        self.lossFunc1 = cfg["lossFunc1"]  # "nn.SmoothL1Loss().to(device)"
        self.lossFunc1Epochs = cfg["lossFunc1Epochs"]  # the epoch number of using lossFunc1

        # Proximal IPM Optimization
        self.useProxialIPM = cfg['useProxialIPM']
        if self.useProxialIPM:
            self.learningStepIPM = cfg['learningStepIPM']  # 0.1
            self.maxIterationIPM = cfg['maxIterationIPM']  # : 50
            self.criterionIPM = cfg['criterionIPM']

        self.useDynamicProgramming = cfg['useDynamicProgramming']
        self.usePrimalDualIPM = cfg['usePrimalDualIPM']
        self.useCEReplaceKLDiv = cfg['useCEReplaceKLDiv']
        self.useLayerDice = cfg['useLayerDice']
        self.useReferSurfaceFromLayer = cfg['useReferSurfaceFromLayer']
        self.useSmoothSurface = cfg['useSmoothSurface']
        self.useWeightedDivLoss = cfg['useWeightedDivLoss']

    def printTo(self, fileHandle):
        fileHandle.write(f"\n=============== Start of Config  ===========\n")
        [fileHandle.write(f"{key}:{value}\n") for key, value in vars(self).items()]
        fileHandle.write(f"=============== End of Config  ===========\n\n")