# Generate Thickness on Validation data

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import random

import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
from network.OCTDataSet import OCTDataSet
from network.OCTOptimization import *
from network.OCTTransform import *


sys.path.append(".")
from ThicknessSubnet_Z4 import ThicknessSubnet_Z4

import time
import numpy as np
import datetime

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.NetTools import *

datasetName =   "test"            # "validation"  or "test"

def printUsage(argv):
    print("============ Generate thickness on Validation data =============")
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

    if datasetName == "validation":
        outputDir = hps.validationOutputDir
    elif datasetName == "test":
        outputDir = hps.testOutputDir
    else:
        assert False

    if hps.dataIn1Parcel:
        if -1 == hps.k and 0 == hps.K:  # do not use cross validation
            testImagesPath = os.path.join(hps.dataDir, datasetName, f"images.npy")
            testLabelsPath = os.path.join(hps.dataDir, datasetName, f"surfaces.npy") if hps.existGTLabel else None
            testIDPath = os.path.join(hps.dataDir, datasetName, f"patientID.json")
        else:  # use cross validation
            testImagesPath = os.path.join(hps.dataDir, datasetName, f"images_CV{hps.k:d}.npy")
            testLabelsPath = os.path.join(hps.dataDir,datasetName, f"surfaces_CV{hps.k:d}.npy")
            testIDPath = os.path.join(hps.dataDir, datasetName, f"patientID_CV{hps.k:d}.json")
    else:
        if -1 == hps.k and 0 == hps.K:  # do not use cross validation
            testImagesPath = os.path.join(hps.dataDir, datasetName, f"patientList.txt")
            testLabelsPath = None
            testIDPath = None
        else:
            print(f"Current do not support Cross Validation and not dataIn1Parcel\n")
            assert (False)

    testData = OCTDataSet(testImagesPath, testIDPath, testLabelsPath,  transform=None, hps=hps)

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("test")

    # test
    testStartTime = time.time()
    net.eval()
    with torch.no_grad():
        testBatch = 0
        net.setStatus("test")
        net.m_epoch = net.m_runParametersDict['epoch']
        for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
            testBatch += 1
            R, loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs=batchData['GTs'],
                                        layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
            if hps.existGTLabel:
                testGts = torch.cat((testGts, batchData['riftWidth'])) if testBatch != 1 else batchData['riftWidth']
            else:
                testGts = None

            testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for future output predict images

            testR = torch.cat((testR, R)) if testBatch != 1 else R


        if hps.existGTLabel:
            goodBScansInGtOrder = None
            stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(testR,
                                                                                                     testGts,
                                                                                                     slicesPerPatient=hps.slicesPerPatient,
                                                                                                     hPixelSize=hps.hPixelSize,
                                                                                                     goodBScansInGtOrder=goodBScansInGtOrder)
            testGts = testGts.cpu().numpy()
            testGTPath = os.path.join(outputDir, f"{datasetName}_thicknessGT_{hps.numSurfaces}surfaces.npy")
            np.save(testGTPath, testGts)

        testR = testR.cpu().numpy()
        testRFilePath = os.path.join(outputDir, f"{datasetName}_result_{hps.numSurfaces}surfaces.npy")
        np.save(testRFilePath, testR)



        # output testID
        with open(os.path.join(outputDir, f"{datasetName}ID.txt"), "w") as file:
            for id in testIDs:
                file.write(f"{id}\n")



    testEndTime = time.time()
    B,S,W = testR.shape

    if hps.existGTLabel:  # compute hausdorff distance
        hausdorffD = columnHausdorffDist(testR, testGts).reshape(1, S)

    # final output result:
    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

    with open(os.path.join(outputDir, f"output_{datasetName}_{timeStr}.txt"), "w") as file:
        hps.printTo(file)
        file.write("\n=======net running parameters=========\n")
        file.write(f"B,S,W = {B, S, W}\n")
        file.write(f"Test time: {testEndTime - testStartTime} seconds.\n")
        file.write(f"net.m_runParametersDict:\n")
        [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]

        file.write(f"\n\n===============Formal Output Result ===========\n")
        if hps.existGTLabel:
            file.write(f"stdSurfaceError = {stdSurfaceError}\n")
            file.write(f"muSurfaceError = {muSurfaceError}\n")
            file.write(f"stdError = {stdError}\n")
            file.write(f"muError = {muError}\n")
            file.write(f"hausdorff Distance = {hausdorffD}\n")



    print(f"============ End of Cross valiation test for OCT Multisurface Network: {hps.experimentName} ===========")




if __name__ == "__main__":
    main()
