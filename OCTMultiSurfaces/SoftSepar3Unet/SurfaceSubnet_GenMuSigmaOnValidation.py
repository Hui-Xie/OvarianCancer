
#Generate Mu and Simga for all validation data.

import sys
import yaml

import torch
import torch.nn as nn
from torch.utils import data


sys.path.append("..")
from network.OCTDataSet import *
from network.OCTOptimization import *
from network.OCTTransform import *
import time

from SurfaceSubnet import SurfaceSubnet

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader

import numpy as np
import datetime

sys.path.append("../dataPrepare_Tongren")
from TongrenFileUtilities import *


def printUsage(argv):
    print("============ Generate Mu and Sigma on Validation data =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    outputXmlSegFiles = True


    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")
    assert "IVUS" not in hps.experimentName

    validationOuputDir = os.path.join(hps.outputDir,"validation")
    if not os.path.exists(validationOuputDir):
        os.makedirs(validationOuputDir)  # recursive dir creation
    validationXmlOuputDir = os.path.join(validationOuputDir, "xml")
    if not os.path.exists(validationXmlOuputDir):
        os.makedirs(validationXmlOuputDir)  # recursive dir creation

    # Only use validation data
    if hps.dataIn1Parcel:
        if -1==hps.k and 0==hps.K:  # do not use cross validation
            testImagesPath = os.path.join(hps.dataDir, "validation", f"images.npy")
            testLabelsPath = os.path.join(hps.dataDir, "validation", f"surfaces.npy") if hps.existGTLabel else None
            testIDPath = os.path.join(hps.dataDir, "validation", f"patientID.json")
        else:  # use cross validation
            testImagesPath = os.path.join(hps.dataDir,"validation", f"images_CV{hps.k:d}.npy")
            testLabelsPath = os.path.join(hps.dataDir,"validation", f"surfaces_CV{hps.k:d}.npy")
            testIDPath    = os.path.join(hps.dataDir,"validation", f"patientID_CV{hps.k:d}.json")
    else:
        if -1==hps.k and 0==hps.K:  # do not use cross validation
            testImagesPath = os.path.join(hps.dataDir, "validation", f"patientList.txt")
            testLabelsPath = None
            testIDPath = None
        else:
            print(f"Current do not support Cross Validation and not dataIn1Parcel\n")
            assert(False)

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


    if "OCT_Tongren" in hps.dataDir:
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
        testBatch = 0
        net.setStatus("test")
        for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
            testBatch += 1
            # S is surface location in (B,S,W) dimension, the predicted Mu
            S, sigma2, _loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
            testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
            testSigma2 = torch.cat((testSigma2, sigma2)) if testBatch != 1 else sigma2
            if hps.existGTLabel:
                testGts = torch.cat((testGts, batchData['GTs'])) if testBatch != 1 else batchData['GTs']
            else:
                testGts = None

            testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for future output predict images

        #output testID
        with open(os.path.join(validationOuputDir, f"validationID.txt"), "w") as file:
            for id in testIDs:
                file.write(f"{id}\n")

        if hps.existGTLabel:
            stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(testOutputs,
                                                                                                     testGts,
                                                                                                     slicesPerPatient=hps.slicesPerPatient,
                                                                                                     hPixelSize=hps.hPixelSize,
                                                                                                     goodBScansInGtOrder=None)
            testGts = testGts.cpu().numpy()
            np.save(os.path.join(validationOuputDir, "validation_gt.npy"), testGts)

        testOutputs = testOutputs.cpu().numpy()
        testSigma2 = testSigma2.cpu().numpy()

        np.save(os.path.join(validationOuputDir,"validation_mu.npy"), testOutputs)
        np.save(os.path.join(validationOuputDir,"validation_simga2.npy"), testSigma2)

        if outputXmlSegFiles:
            batchPrediciton2OCTExplorerXML(testOutputs, testIDs, hps.slicesPerPatient, surfaceNames, validationXmlOuputDir,
                                           y=hps.inputHeight, voxelSizeY=hps.hPixelSize, dataInSlice=hps.dataInSlice)

    testEndTime = time.time()

    # check testOutputs whether violate surface-separation constraints
    testOutputs0 = testOutputs[:,0:-1,:]
    testOutputs1 = testOutputs[:, 1:, :]
    violateConstraintErrors = np.nonzero(testOutputs0 > testOutputs1)

    # final output result:
    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

    B,S, W = testOutputs.shape

    with open(os.path.join(validationOuputDir,f"ValidationOutput_{timeStr}.txt"), "w") as file:
        hps.printTo(file)
        file.write("\n=======net running parameters=========\n")
        file.write(f"B,S,W = {B, S, W}\n")
        file.write(f"Test time: {testEndTime-testStartTime} seconds.\n")
        file.write(f"net.m_runParametersDict:\n")
        [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]

        file.write(f"\n\n===============Formal Output Result ===========\n")
        if hps.existGTLabel:
            file.write(f"stdSurfaceError = {stdSurfaceError}\n")
            file.write(f"muSurfaceError = {muSurfaceError}\n")
            file.write(f"stdError = {stdError}\n")
            file.write(f"muError = {muError}\n")
        file.write(f"pixel number of violating surface-separation constraints: {len(violateConstraintErrors[0])}\n")

        if 0 != len(violateConstraintErrors[0]):
            violateConstraintSlices = set(violateConstraintErrors[0])
            file.write(f"slice number of violating surface-separation constraints: {len(violateConstraintSlices)}\n")
            file.write("slice list of violating surface-separation constraints:\n")
            for s in violateConstraintSlices:
                file.write(f"\t{testIDs[s]}\n")



    print(f"============ End of generating mu and sigma2 for  OCT Multisurface Network: {hps.experimentName} ===========")


if __name__ == "__main__":
    main()