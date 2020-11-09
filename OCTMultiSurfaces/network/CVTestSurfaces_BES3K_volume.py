
# Test BES_3K volume by volume and output its latent vector.

import sys
import yaml

import torch
import torch.nn as nn
from torch.utils import data


sys.path.append(".")
from OCTDataSetVolume import OCTDataSetVolume
from SurfacesUnet_BES3K0512 import SurfacesUnet_BES3K0512
from OCTOptimization import *
from OCTTransform import *
import time

sys.path.append("..")
sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader

import numpy as np
import datetime

sys.path.append("../dataPrepare_Tongren")
from TongrenFileUtilities import saveNumpy2OCTExplorerXML


def printUsage(argv):
    print("============ Test OCT BES_3K MultiSurface Network with input volume by volume =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1


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
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # Load network

    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("test")

    testImagesPath = hps.dataDir
    testLabelsPath = None
    testIDPath    = None

    testData = OCTDataSetVolume(testImagesPath, testIDPath, testLabelsPath,  transform=None, hps=hps)

    # test
    testStartTime = time.time()
    net.eval()
    with torch.no_grad():
        testBatch = 0
        for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
            testBatch += 1

            # squeeze the extra dimension of data with volume input
            inputs = batchData['images'].squeeze(dim=0)
            volumeID = batchData['IDs'][0]  # erase tuple wrapper

            # S is surface location in (B,S,W) dimension, the predicted Mu
            S, _loss, latentVector = net.forward(inputs, gaussianGTs=None, GTs = None, layerGTs=None)

            # Error Std and mean
            predicition = S
            if hps.groundTruthInteger:
                predicition = (S + 0.5).int()  # as ground truth are integer, make the output also integers.

            predicition = predicition.cpu().numpy()

            outputXMLFilename = hps.xmlOutputDir2 + f"/{volumeID}_Sequence_Surfaces_Prediction.xml"
            if hps.outputXmlSegFiles and (not os.path.exists(outputXMLFilename)):
                saveNumpy2OCTExplorerXML(volumeID, predicition, surfaceNames, hps.xmlOutputDir2, hps.refXMLFile, y=hps.inputHeight, voxelSizeY=hps.hPixelSize)

            outputLatentPath = hps.latentDir + f"/{volumeID}_latent.npy"
            if hps.outputLatent and (not os.path.exists(outputLatentPath)):
                np.save(outputLatentPath, latentVector.cpu().numpy())

    testEndTime = time.time()

    # final output result:
    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

    with open(os.path.join(hps.outputDir,f"output_Volume_{timeStr}.txt"), "w") as file:
        hps.printTo(file)
        file.write("\n=======net running parameters=========\n")
        file.write(f"Test time: {testEndTime-testStartTime} seconds.\n")
        file.write(f"net.m_runParametersDict:\n")
        [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]

    print(f"============ End of BES3K test for OCT Multisurface Network: {hps.experimentName} ===========")


if __name__ == "__main__":
    main()