
# Test BES_3K 200 packages data.

import sys
import yaml

import torch
import torch.nn as nn
from torch.utils import data


sys.path.append(".")
from OCTDataSet import *
from SurfacesUnet import SurfacesUnet
from SurfacesUnet_YufanHe import SurfacesUnet_YufanHe
from OCTOptimization import *
from OCTTransform import *
import time

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader

import matplotlib.pyplot as plt
import numpy as np
import datetime

sys.path.append("../dataPrepare_Tongren")
from TongrenFileUtilities import *


def printUsage(argv):
    print("============ Test OCT BES_3K MultiSurface Network =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # output config
    MarkGTDisorder = False
    MarkPredictDisorder = False

    outputXmlSegFiles = True

    OutputNumImages = 0
    # choose from 0, 1,2,3:----------
    # 0: no image output; 1: Prediction; 2: GT and Prediction; 3: Raw, GT, Prediction
    # 4: Raw, GT, Prediction with GT superpose in one image
    comparisonSurfaceIndex = None
    #comparisonSurfaceIndex = 2 # compare the surface 2 (index starts from  0)
    # GT uses red, while prediction uses green

    needLegend = True

    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")
    assert "IVUS" not in hps.experimentName

    if hps.numSurfaces == 9:
        surfaceNames = ['ILM', 'RNFL-GCL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE', 'OB_RPE']
    if hps.numSurfaces == 10:
        surfaceNames = ['ILM', 'RNFL-GCL', 'GCL-IPL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE', 'OB_RPE']

    # construct network
    net = eval(hps.network)(hps.inputHight, hps.inputWidth, inputChannels=hps.inputChannels, nLayers=hps.nLayers,
                            numSurfaces=hps.numSurfaces, N=hps.startFilters)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # Load network
    if os.path.exists(hps.netPath) and len(getFilesList(hps.netPath, ".pt")) >= 2:
        netMgr = NetMgr(net, hps.netPath, hps.device)
        netMgr.loadNet("test")
        print(f"Network load from  {hps.netPath}")
    else:
        print(f"Can not find pretrained network for test!")

    net.hps = hps

    GPUIndex = int(hps.GPUIndex)

    for k in range(GPUIndex, hps.K, hps.N_GPU):
        testImagesPath = os.path.join(hps.dataDir,"test", f"images_{k}.npy")
        testLabelsPath = None
        testIDPath    = os.path.join(hps.dataDir,"test", f"patientID_{k}.json")

        testData = OCTDataSet(testImagesPath, testIDPath, testLabelsPath,  transform=None, device=hps.device, sigma=hps.sigma,
                          lacingWidth=hps.lacingWidth, TTA=False, TTA_Degree=0, scaleNumerator=hps.scaleNumerator,
                          scaleDenominator=hps.scaleDenominator, gradChannels=hps.gradChannels)

        # test
        testStartTime = time.time()
        net.eval()
        with torch.no_grad():
            testBatch = 0
            for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
                testBatch += 1
                # S is surface location in (B,S,W) dimension, the predicted Mu
                S, _loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'])

                batchImages = batchData['images'][:, 0, :, :]  # erase grad channels to save memory
                images = torch.cat((images, batchImages)) if testBatch != 1 else batchImages # for output result
                testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
                testGts = None

                testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for future output predict images


            # Error Std and mean
            if hps.groundTruthInteger:
                testOutputs = (testOutputs + 0.5).int()  # as ground truth are integer, make the output also integers.

            images = images.cpu().numpy().squeeze()
            testOutputs = testOutputs.cpu().numpy()

            if outputXmlSegFiles:
                batchPrediciton2OCTExplorerXML(testOutputs, testIDs, hps.slicesPerPatient, surfaceNames, hps.xmlOutputDir)

        testEndTime = time.time()
        B, H, W = images.shape
        B, S, W = testOutputs.shape
        patientIDList = []



        # check testOutputs whether violate surface-separation constraints
        testOutputs0 = testOutputs[:,0:-1,:]
        testOutputs1 = testOutputs[:, 1:, :]
        violateConstraintErrors = np.nonzero(testOutputs0 > testOutputs1)

        # final output result:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        with open(os.path.join(hps.outputDir,f"output_pacakage_{k}_{timeStr}.txt"), "w") as file:
            hps.printTo(file)
            file.write("\n=======net running parameters=========\n")
            file.write(f"B,S,H,W = {B, S, H, W}\n")
            file.write(f"Test time: {testEndTime-testStartTime} seconds.\n")
            file.write(f"net.m_runParametersDict:\n")
            [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]

            file.write(f"\n\n===============Formal Output Result ===========\n")
            file.write(f"patientIDList ={patientIDList}\n")
            file.write(f"pixel number of violating surface-separation constraints: {len(violateConstraintErrors[0])}\n")

            if 0 != len(violateConstraintErrors[0]):
                violateConstraintSlices = set(violateConstraintErrors[0])
                file.write(f"slice number of violating surface-separation constraints: {len(violateConstraintSlices)}\n")
                file.write("slice list of violating surface-separation constraints:\n")
                for s in violateConstraintSlices:
                    file.write(f"\t{testIDs[s]}\n")



    print(f"============ End of BES3K test for OCT Multisurface Network: {hps.experimentName} ===========")


if __name__ == "__main__":
    main()