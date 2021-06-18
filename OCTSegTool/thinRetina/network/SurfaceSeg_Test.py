
# Cross Validation Test

import sys
import os
import torch
from torch.utils import data


sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")


from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.SurfaceSegNet_Q import SurfaceSegNet_Q
from OCTData.OCTDataSet import  OCTDataSet
from OCTData.OCTDataUtilities import computeMASDError, batchPrediciton2OCTExplorerXML, outputNumpyImagesSegs
from framework.NetTools import columnHausdorffDist

import time
import matplotlib.pyplot as plt
import numpy as np
import datetime

def printUsage(argv):
    print("============ Cross Validation Test OCT MultiSurface Network =============")
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

    OutputNumImages = 3
    # choose from 0, 1,2,3:----------
    # 0: no image output; 1: Prediction; 2: GT and Prediction; 3: Raw, GT, Prediction
    # 4: Raw, GT, Prediction with GT superpose in one image
    comparisonSurfaceIndex = None
    #comparisonSurfaceIndex = 2 # compare the surface 2 (index starts from  0)
    # GT uses red, while prediction uses green

    OnlyOutputGoodBscans =False
    needLegend = True

    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")
    assert "IVUS" not in hps.experimentName

    if hps.dataIn1Parcel:
        if -1==hps.k and 0==hps.K:  # do not use cross validation
            testImagesPath = os.path.join(hps.dataDir, "test", f"images.npy")
            testLabelsPath = os.path.join(hps.dataDir, "test", f"surfaces.npy") if hps.existGTLabel else None
            testIDPath = os.path.join(hps.dataDir, "test", f"patientID.json")
        else:  # use cross validation
            testImagesPath = os.path.join(hps.dataDir,"test", f"images_CV{hps.k:d}.npy")
            testLabelsPath = os.path.join(hps.dataDir,"test", f"surfaces_CV{hps.k:d}.npy")
            testIDPath    = os.path.join(hps.dataDir,"test", f"patientID_CV{hps.k:d}.json")
    else:
        if -1==hps.k and 0==hps.K:  # do not use cross validation
            testImagesPath = os.path.join(hps.dataDir, "test", f"patientList.txt")
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

    # Specific for different application
    assert hps.numSurfaces ==6
    surfaceNames = ("ILM", "RNFL-GCL", "IPL-INL", "OPL-HFL", "BMEIS", "OB_RPE")
    pltColors = ('tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:red', 'tab:green')

    # test
    testStartTime = time.time()
    net.eval()
    with torch.no_grad():
        testBatch = 0
        net.setStatus("test")
        for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
            testBatch += 1
            # S is surface location in (B,S,W) dimension, the predicted Mu
            S, _sigma2, _loss, _x = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
            batchImages = batchData['images'][:, 0, :, :]  # erase grad channels to save memory
            images = torch.cat((images, batchImages)) if testBatch != 1 else batchImages # for output result
            testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
            sigma2 = torch.cat((sigma2, _sigma2)) if testBatch != 1 else _sigma2
            if hps.existGTLabel:
                testGts = torch.cat((testGts, batchData['GTs'])) if testBatch != 1 else batchData['GTs']
            else:
                testGts = None

            testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for future output predict images

        #output testID
        with open(os.path.join(hps.outputDir, f"testID.txt"), "w") as file:
            for id in testIDs:
                file.write(f"{id}\n")

        if hps.groundTruthInteger:
            testOutputs = (testOutputs + 0.5).int()  # as ground truth are integer, make the output also integers.

        # Specific for different application
        # get volumeIDs and volumeBscanStartIndexList
        volumeIDs = []
        volumeBscanStartIndexList = []
        B = len(testIDs)
        for i in range(0, B):  # we need consider the different Bscan numbers for different volumes.
            id = testIDs[i]
            if '_s000' == id[-5:]:
                volumeIDs.append(id[: id.rfind("_s000")])
                volumeBscanStartIndexList.append(i)

        if hps.existGTLabel: # Error Std and mean
            stdSurfaceError, muSurfaceError, stdError, muError =  computeMASDError(testOutputs, testGts, volumeBscanStartIndexList, hPixelSize=hps.hPixelSize)
            testGts = testGts.cpu().numpy()

        images = images.cpu().numpy().squeeze()
        testOutputs = testOutputs.cpu().numpy()

        if outputXmlSegFiles:
            batchPrediciton2OCTExplorerXML(testOutputs, volumeIDs, volumeBscanStartIndexList, surfaceNames, hps.xmlOutputDir,
                                           refXMLFile=hps.refXMLFile,
                                           penetrationChar=hps.penetrationChar, penetrationPixels=hps.inputHeight, voxelSizeUnit=hps.voxelSizeUnit,
                                           voxelSizeX=hps.voxelSizeX, voxelSizeY=hps.voxelSizeY, voxelSizeZ=hps.voxelSizeZ)
            outputNumpyImagesSegs(images, testOutputs, volumeIDs, volumeBscanStartIndexList, hps.testOutputDir)


    testEndTime = time.time()

    #generate predicted images
    B,H,W = images.shape
    B, S, W = testOutputs.shape
    patientIDList = []

    if hps.existGTLabel:  # compute hausdorff distance
        hausdorffD = columnHausdorffDist(testOutputs, testGts).reshape(1, S)

    # check testOutputs whether violate surface-separation constraints
    testOutputs0 = testOutputs[:, 0:-1, :]
    testOutputs1 = testOutputs[:, 1:, :]
    violateConstraintErrors = np.nonzero(testOutputs0 > testOutputs1)

    # final output result:
    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

    with open(os.path.join(hps.outputDir, f"output_{timeStr}.txt"), "w") as file:
        hps.printTo(file)
        file.write("\n=======net running parameters=========\n")
        file.write(f"B,S,H,W = {B, S, H, W}\n")
        file.write(f"Test time: {testEndTime - testStartTime} seconds.\n")
        file.write(f"net.m_runParametersDict:\n")
        [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]

        file.write(f"\n\n===============Formal Output Result ===========\n")
        file.write(f"patientIDList ={patientIDList}\n")
        if hps.existGTLabel:
            file.write(f"stdSurfaceError = {stdSurfaceError}\n")
            file.write(f"muSurfaceError = {muSurfaceError}\n")
            file.write(f"stdError = {stdError}\n")
            file.write(f"muError = {muError}\n")
            file.write(f"hausdorff Distance in pixels = {hausdorffD}\n")

        file.write(f"pixel number of violating surface-separation constraints: {len(violateConstraintErrors[0])}\n")

        if 0 != len(violateConstraintErrors[0]):
            violateConstraintSlices = set(violateConstraintErrors[0])
            file.write(f"slice number of violating surface-separation constraints: {len(violateConstraintSlices)}\n")
            file.write("slice list of violating surface-separation constraints:\n")
            for s in violateConstraintSlices:
                file.write(f"\t{testIDs[s]}\n")

    # output images
    assert S <= len(pltColors)

    for b in range(B):
        patientID_Index = testIDs[b]
        if OutputNumImages ==0:
            continue

        f = plt.figure(frameon=False)
        # DPI = f.dpi
        DPI = 100.0

        assert OutputNumImages ==3
        rowSubplot = 1
        colSubplot = 3
        imageFileName = patientID_Index + "_Raw_GT_Predict.png"
        f.set_size_inches(W * colSubplot / float(DPI), H * rowSubplot / float(DPI))

        plt.margins(0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

        subplot1 = plt.subplot(rowSubplot, colSubplot, 1)
        subplot1.imshow(images[b, :, :], cmap='gray')
        subplot1.axis('off')

        subplot2 = plt.subplot(rowSubplot, colSubplot, 2)
        subplot2.imshow(images[b, :, :], cmap='gray')
        for s in range(0, S):
            subplot2.plot(range(0, W), testGts[b, s, :], pltColors[s], linewidth=1.2)
        if needLegend:
            subplot2.legend(surfaceNames, loc='lower left', ncol=2, fontsize='x-small')
        subplot2.axis('off')

        subplot3 = plt.subplot(rowSubplot, colSubplot, 3)
        subplot3.imshow(images[b, :, :], cmap='gray')
        for s in range(0, S):
            subplot3.plot(range(0, W), testOutputs[b, s, :], pltColors[s], linewidth=1.2)
        if needLegend:
            subplot3.legend(surfaceNames, loc='lower left', ncol=2, fontsize='x-small')
        subplot3.axis('off')

        plt.savefig(os.path.join(hps.imagesOutputDir,imageFileName), dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"============ End of Cross valiation test for OCT Multisurface Network: {hps.experimentName} ===========")


if __name__ == "__main__":
    main()