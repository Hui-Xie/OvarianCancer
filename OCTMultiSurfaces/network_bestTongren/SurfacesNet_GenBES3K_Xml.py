
# Use best Tongren Network to generate BES_3K all xml segmentation result

import sys
import yaml

import torch
import torch.nn as nn
from torch.utils import data


sys.path.append("..")
from network.OCTDataSetVolume import *
from network.OCTOptimization import *
from network.OCTTransform import *
import time

from SurfacesNet import SurfacesNet

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
    print("============ Generate all xml segmentation of BES_3K data =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # output config
    outputXmlSegFiles = True

    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")
    assert "IVUS" not in hps.experimentName

    testImagesPath =  hps.dataDir           # os.path.join(hps.dataDir, f"patientList.txt")
    testLabelsPath = None
    testIDPath = None

    testData = OCTDataSetVolume(testImagesPath, testIDPath, testLabelsPath,  transform=None, hps=hps)

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("test")


    if ("OCT_Tongren" in hps.dataDir) or ("BES_3K" in hps.dataDir) :
        if hps.numSurfaces == 9:
            surfaceNames = ['ILM', 'RNFL-GCL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE', 'OB_RPE']
        if hps.numSurfaces == 10:
            surfaceNames = ['ILM', 'RNFL-GCL', 'GCL-IPL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE', 'OB_RPE']
    if "OCT_JHU" in hps.dataDir:
        surfaceNames = ['ILM', 'RNFL-GCL', 'IPL-INL', 'INL-OPL', 'OPL-ONL', 'ELM', 'IS-OS', 'OS-RPE', 'BM']

    if "OCT_Duke" in hps.dataDir:
        if hps.numSurfaces == 3:  # for Duke data
            surfaceNames = ['ILM', 'InterRPEDC', 'OBM']

    # test
    testStartTime = time.time()
    net.eval()
    with torch.no_grad():
        net.setStatus("test")
        for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):

            # squeeze the extra dimension of data with volume input
            inputs = batchData['images'].squeeze(dim=0)
            volumeID = batchData['IDs'][0]  # erase tuple wrapper

            # S is surface location in (B,S,W) dimension, the predicted Mu
            S, _sigma2, _loss = net.forward(inputs, gaussianGTs=None, GTs = None, layerGTs=None, riftGTs=None)
            if hps.groundTruthInteger:
                S = (S + 0.5).int()  # as ground truth are integer, make the output also integers.
            testOutput = S.cpu().numpy()
            if outputXmlSegFiles:
                saveNumpy2OCTExplorerXML(volumeID, testOutput, surfaceNames, hps.xmlOutputDir, hps.refXMLFile, y=hps.inputHeight,
                                         voxelSizeY=hps.hPixelSize)
            break

    testEndTime = time.time()


    # final output result:
    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

    with open(os.path.join(hps.outputDir,f"test_output_{timeStr}.txt"), "w") as file:
        hps.printTo(file)
        file.write("\n=======net running parameters=========\n")
        file.write(f"Test time: {testEndTime-testStartTime} seconds.\n")
        file.write(f"net.m_runParametersDict:\n")
        [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]

        file.write(f"\n\n===============Formal Output Result ===========\n")
    print(f"============ End of generate xml segmentation of BES_3K: {hps.experimentName} ===========")


if __name__ == "__main__":
    main()