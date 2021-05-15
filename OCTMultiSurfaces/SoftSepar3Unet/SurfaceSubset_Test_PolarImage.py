
# Test output of Polar image segmentation.

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
from SurfaceSubnet_M import SurfaceSubnet_M  # smooth module surface net
from SurfacesUnet_YufanHe_2 import SurfacesUnet_YufanHe_2
from SurfaceSubnet_M5 import SurfaceSubnet_M5
from SurfaceSubnet_P import SurfaceSubnet_P
from SurfaceSubnet_Q import SurfaceSubnet_Q

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.NetTools import *

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from numpy import genfromtxt
import datetime

# import matlab.engine

sys.path.append("../dataPrepare_IVUS")
from PolarCartesianConverter import PolarCartesianConverter

def printUsage(argv):
    print("============ Cross Validation Test OCT MultiSurface Network for Polar images =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # debug:
    MarkGTDisorder = False
    MarkPredictDisorder = False
    OutputPredictImages = False
    Output2Images = False  # False means output 3 images.

    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")
    assert "IVUS" in hps.experimentName


    if -1==hps.k and 0==hps.K:  # do not use cross validation
        testImagesPath = os.path.join(hps.dataDir, "test", f"images.npy")
        testLabelsPath = os.path.join(hps.dataDir, "test", f"surfaces.npy")
        testIDPath = os.path.join(hps.dataDir, "test", f"patientID.json")
    else:  # use cross validation
        testImagesPath = os.path.join(hps.dataDir,"test", f"images_CV{hps.k:d}.npy")
        testLabelsPath = os.path.join(hps.dataDir,"test", f"surfaces_CV{hps.k:d}.npy")
        testIDPath    = os.path.join(hps.dataDir,"test", f"patientID_CV{hps.k:d}.json")

    testData = OCTDataSet(testImagesPath, testIDPath, testLabelsPath, transform=None, hps=hps)

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("test")

    surfaceNames = ['lumen', 'media']

    # test
    net.eval()
    with torch.no_grad():
        testBatch = 0
        net.setStatus("test")
        for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
            testBatch += 1
            # S is surface location in (B,S,W) dimension, the predicted Mu
            S, _sigma2, _loss, _x = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'],
                                                GTs=batchData['GTs'], layerGTs=batchData['layers'],
                                                riftGTs=batchData['riftWidth'])
            batchImages = batchData['images'][:, 0, :, :]  # erase grad channels to save memory
            images = torch.cat((images, batchImages)) if testBatch != 1 else batchImages  # for output result
            testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
            sigma2 = torch.cat((sigma2, _sigma2)) if testBatch != 1 else _sigma2
            if hps.existGTLabel:
                testGts = torch.cat((testGts, batchData['GTs'])) if testBatch != 1 else batchData['GTs']
            else:
                testGts = None

            testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData[
                'IDs']  # for future output predict images

        if hps.debug == True:
            print(f"sigma2.shape = {sigma2.shape}")
            print(f"mean of sigma2 = {torch.mean(sigma2, dim=[0, 2])}")
            print(f"min of sigma2  = {torch.min(torch.min(sigma2, dim=0)[0], dim=-1)}")
            print(f"max of sigma2  = {torch.max(torch.max(sigma2, dim=0)[0], dim=-1)}")

        # output testID
        with open(os.path.join(hps.outputDir, f"testID.txt"), "w") as file:
            for id in testIDs:
                file.write(f"{id}\n")

        # Error Std and mean
        if hps.groundTruthInteger:
            testOutputs = (testOutputs + 0.5).int()  # as ground truth are integer, make the output also integers.

    stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(testOutputs, testGts,
                                                                                             slicesPerPatient=hps.slicesPerPatient,
                                                                                             hPixelSize=hps.hPixelSize)
    #generate predicted images
    images = images.cpu().numpy().squeeze()
    B,H,W = images.shape
    B, S, W = testOutputs.shape
    testOutputs = testOutputs.cpu().numpy()
    testGts = testGts.cpu().numpy()
    patientIDList = []

    pltColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',  'tab:olive', 'tab:brown', 'tab:pink', 'tab:red', 'tab:cyan']
    assert S <= len(pltColors)

    outputTxtDir = os.path.join(hps.outputDir, "text")
    outputImageDir = os.path.join(hps.outputDir, "images")
    if not os.path.exists(outputTxtDir):
        os.makedirs(outputTxtDir)  # recursive dir creation
    if not os.path.exists(outputImageDir):
        os.makedirs(outputImageDir)  # recursive dir creation

    polarConverter = PolarCartesianConverter((384,384),192, 192, 192, 360)

    # this is original cartesian coordination GT files.
    #  testGTSegDir= "/raid001/users/hxie1/data/IVUS/Test_Set/Data_set_B/LABELS_obs1"
    testGTSegDir = os.path.join(os.path.dirname(hps.dataDir), "Test_Set/Data_set_B/LABELS_obs1")

    # eng = matlab.engine.start_matlab()
    # eng.addpath(r'/local/vol00/scratch/Users/hxie1/Projects/DeepLearningSeg/OCTMultiSurfaces/dataPrepare_IVUS')
    for b in range(B):
        patientID = os.path.splitext(os.path.basename(testIDs[b]))[0]  # frame_05_0004_003

        # output txt predict result
        lumenPredictFile = os.path.join(outputTxtDir, "lum_" + patientID + ".txt")  # lum_frame_01_0001_003.txt
        mediaPredictFile = os.path.join(outputTxtDir, "med_" + patientID + ".txt")  # e.g. med_frame_01_0030_003.txt

        lumenGTFile = os.path.join(testGTSegDir, "lum_" + patientID + ".txt")  # lum_frame_01_0001_003.txt
        mediaGTFile = os.path.join(testGTSegDir, "med_" + patientID + ".txt")  # e.g. med_frame_01_0030_003.txt

        cartesianLabel = polarConverter.polarLabel2Cartesian(testOutputs[b]) # size: C*N*2
        # IVUS demanding format: only one decimal
        np.savetxt(lumenPredictFile, cartesianLabel[0,], fmt="%.1f", delimiter=",")
        np.savetxt(mediaPredictFile, cartesianLabel[1,], fmt="%.1f", delimiter=",")
        #np.savetxt(lumenPredictFile, cartesianLabel[0,], delimiter=",")
        #np.savetxt(mediaPredictFile, cartesianLabel[1,], delimiter=",")


        # output image
        if not OutputPredictImages:
            continue
        cartesianLabel = np.concatenate((cartesianLabel, np.expand_dims(cartesianLabel[:,0,:],axis=1)),axis=1)  # close curve.
        imageb = imread(testIDs[b]).astype(np.float32)

        lumenGTLabel = genfromtxt(lumenGTFile, delimiter=',')
        mediaGTLabel = genfromtxt(mediaGTFile, delimiter=',')

        f = plt.figure(frameon=False)
        # DPI = f.dpi
        DPI = 100
        W=H=384
        if W/H > 16/9:  # normal screen resolution rate is 16:9
            subplotRow = 3
            subplotCol = 1
        else:
            subplotRow = 1
            subplotCol = 3
        if Output2Images:
            subplotRow = 1
            subplotCol = 2
        f.set_size_inches(W * subplotCol / float(DPI), H*subplotRow / float(DPI))

        plt.margins(0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

        subplotIndex = 0

        if not Output2Images:
            subplotIndex += 1
            subplot1 = plt.subplot(subplotRow, subplotCol, subplotIndex)
            subplot1.imshow(imageb, cmap='gray')
            subplot1.axis('off')

        subplotIndex += 1
        subplot2 = plt.subplot(subplotRow, subplotCol, subplotIndex)
        subplot2.imshow(imageb, cmap='gray')
        subplot2.plot(lumenGTLabel[:, 0], lumenGTLabel[:, 1], linewidth=0.9)
        subplot2.plot(mediaGTLabel[:, 0], mediaGTLabel[:, 1], linewidth=0.9)
        subplot2.axis('off')

        subplotIndex += 1
        subplot3 = plt.subplot(subplotRow, subplotCol, subplotIndex)
        subplot3.imshow(imageb, cmap='gray')
        for s in range(0, S):
            subplot3.plot(cartesianLabel[s,:,0], cartesianLabel[s,:,1], linewidth=0.9)
        subplot3.axis('off')


        if Output2Images:
            plt.savefig(os.path.join(outputImageDir, patientID + "_GT_Predict.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(os.path.join(outputImageDir, patientID + "_Image_GT_Predict.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close()

    # check testOutputs whehter violate surface-separation constraints
    testOutputs0 = testOutputs[:,0:-1,:]
    testOutputs1 = testOutputs[:, 1:, :]
    violateConstraintErrors = np.nonzero(testOutputs0 > testOutputs1)

    # final output result:
    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

    with open(os.path.join(hps.outputDir,f"output_{timeStr}.txt"), "w") as file:
        hps.printTo(file)
        file.write("\n=======net running parameters=========\n")
        file.write(f"B,S,H,W = {B,S,H, W}\n")
        file.write(f"net.m_runParametersDict:\n")
        [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]
        file.write(f"\n\n===============Formal Output Result ===========\n")
        file.write(f"stdSurfaceError = {stdSurfaceError}\n")
        file.write(f"muSurfaceError = {muSurfaceError}\n")
        file.write(f"patientIDList ={patientIDList}\n")
        #file.write(f"stdPatientError = {stdPatientError}\n")
        #file.write(f"muPatientError = {muPatientError}\n")
        file.write(f"stdError = {stdError}\n")
        file.write(f"muError = {muError}\n")
        file.write(f"pixel number of violating surface-separation constraints: {len(violateConstraintErrors[0])}\n")
        if 0 != len(violateConstraintErrors[0]):
            violateConstraintSlices = set(violateConstraintErrors[0])
            file.write(f"slice number of violating surface-separation constraints: {len(violateConstraintSlices)}\n")
            file.write("slice list of violating surface-separation constraints:\n")
            for s in violateConstraintSlices:
                file.write(f"\t{testIDs[s]}\n")



    print(f"============ End of Test: {hps.experimentName} ===========")


if __name__ == "__main__":
    main()