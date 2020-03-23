
# Cross Validation Test

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
    OutputPredictImages = False
    outputXmlSegFiles = True
    Output2Images = True
    needLegend = True


    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")
    assert "IVUS" not in hps.experimentName

    if -1==hps.k and 0==hps.K:  # do not use cross validation
        testImagesPath = os.path.join(hps.dataDir, "test", f"images.npy")
        testLabelsPath = os.path.join(hps.dataDir, "test", f"surfaces.npy") if hps.existGTLabel else None
        testIDPath = os.path.join(hps.dataDir, "test", f"patientID.json")
    else:  # use cross validation
        testImagesPath = os.path.join(hps.dataDir,"test", f"images_CV{hps.k:d}.npy")
        testLabelsPath = os.path.join(hps.dataDir,"test", f"surfaces_CV{hps.k:d}.npy")
        testIDPath    = os.path.join(hps.dataDir,"test", f"patientID_CV{hps.k:d}.json")

    testData = OCTDataSet(testImagesPath, testIDPath, testLabelsPath,  transform=None, device=hps.device, sigma=hps.sigma,
                          lacingWidth=hps.lacingWidth, TTA=False, TTA_Degree=0, scaleNumerator=hps.scaleNumerator,
                          scaleDenominator=hps.scaleDenominator, gradChannels=hps.gradChannels)

    # construct network
    net = eval(hps.network)(hps.inputHight, hps.inputWidth, inputChannels=hps.inputChannels, nLayers=hps.nLayers,
                        numSurfaces=hps.numSurfaces, N=hps.numStartFilters)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)


    # Load network
    if os.path.exists(hps.netPath) and len(getFilesList(hps.netPath, ".pt")) >= 2 :
        netMgr = NetMgr(net, hps.netPath, hps.device)
        netMgr.loadNet("test")
        print(f"Network load from  {hps.netPath}")
    else:
        print(f"Can not find pretrained network for test!")

    net.hps = hps

    # test
    net.eval()
    with torch.no_grad():
        testBatch = 0
        for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
            testBatch += 1
            # S is surface location in (B,S,W) dimension, the predicted Mu
            S, _loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'])

            images = torch.cat((images, batchData['images'])) if testBatch != 1 else batchData['images'] # for output result
            testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
            testGts = torch.cat((testGts, batchData['GTs'])) if testBatch != 1 else batchData['GTs'] if hps.existGTLabel else None
            testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for future output predict images


        # Error Std and mean
        if hps.groundTruthInteger:
            testOutputs = (testOutputs + 0.5).int()  # as ground truth are integer, make the output also integers.

        if hps.existGTLabel:
            stdSurfaceError, muSurfaceError, stdError, muError  = computeErrorStdMuOverPatientDimMean(testOutputs, testGts,
                                                                                  slicesPerPatient=hps.slicesPerPatient,
                                                                                  hPixelSize=hps.hPixelSize)

    #generate predicted images
    images = images[:, 0, :, :]  # erase grad channels
    images = images.cpu().numpy().squeeze()
    B,H,W = images.shape
    B, S, W = testOutputs.shape
    testOutputs = testOutputs.cpu().numpy()
    testGts = testGts.cpu().numpy()
    patientIDList = []

    pltColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',  'tab:olive', 'tab:brown', 'tab:pink', 'tab:red', 'tab:cyan']
    assert S <= len(pltColors)

    for b in range(B):
        if "OCT_Tongren" in hps.dataDir:
            # example: "/home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
            patientID_Index = extractFileName(testIDs[b])  #e.g.: 4511_OD_29134_OCT06
            if "_OCT01" in patientID_Index:
                patientIDList.append(extractPaitentID(testIDs[b]))
            surfaceNames = ['ILM', 'RNFL-GCL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE', 'OB_RPE']
        if "OCT_JHU" in hps.dataDir:
            # testIDs[0] = '/home/hxie1/data/OCT_JHU/preprocessedData/image/hc01_spectralis_macula_v1_s1_R_19.png'
            patientID_Index = os.path.splitext(os.path.basename(testIDs[b]))[0]  #e.g. hc01_spectralis_macula_v1_s1_R_19
            patient = patientID_Index[0:4] # e.g. hc01
            if "_s1_R_19" in patientID_Index and patient not in patientIDList:
                patientIDList.append(patient)
            surfaceNames = ['ILM', 'RNFL-GCL', 'IPL-INL', 'INL-OPL', 'OPL-ONL', 'ELM', 'IS-OS', 'OS-RPE', 'BM']


        if not OutputPredictImages:
            continue

        f = plt.figure(frameon=False)
        DPI = f.dpi

        if Output2Images:
            if "OCT_Tongren" in hps.dataDir:
                subplotRow = 1
                subplotCol = 2
            else:
                subplotRow = 2
                subplotCol = 1
            f.set_size_inches(W*subplotCol / float(DPI), H * subplotRow / float(DPI))
        else:
            if W/H > 16/9:  # normal screen resolution rate is 16:9
                f.set_size_inches(W/ float(DPI), H*3 / float(DPI))
                subplotRow = 3
                subplotCol = 1
            else:
                f.set_size_inches(W*3/float(DPI), H/float(DPI))
                subplotRow = 1
                subplotCol = 3

        plt.margins(0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

        subplotIndex = 0

        if not Output2Images:
            subplotIndex += 1

            subplot1 = plt.subplot(subplotRow, subplotCol, subplotIndex)
            subplot1.imshow(images[b,].squeeze(), cmap='gray')
            if MarkGTDisorder:
                gt0 = testGts[b, 0:-1, :]
                gt1 = testGts[b, 1:,   :]
                errorLocations = np.nonzero(gt0>gt1)  # return as tuple
                if len(errorLocations[0]) > 0:
                    subplot1.scatter(errorLocations[1], testGts[b, errorLocations[0], errorLocations[1]], s=1, c='r', marker='o')  # red for gt disorder
            if MarkPredictDisorder:
                predict0 = testOutputs[b, 0:-1, :]
                predict1 = testOutputs[b, 1:,   :]
                errorLocations = np.nonzero(predict0 > predict1)  # return as tuple
                if len(errorLocations[0]) > 0:
                    subplot1.scatter(errorLocations[1], testOutputs[b, errorLocations[0], errorLocations[1]], s=1, c='g', marker='o') # green for prediction disorder
            subplot1.axis('off')

        subplotIndex += 1
        subplot2 = plt.subplot(subplotRow, subplotCol, subplotIndex)
        subplot2.imshow(images[b,].squeeze(), cmap='gray')
        for s in range(0, S):
            subplot2.plot(range(0, W), testGts[b, s, :].squeeze(), pltColors[s], linewidth=0.9)
        if needLegend:
            if "OCT_Tongren" in hps.dataDir:
                subplot2.legend(surfaceNames, loc='lower center', ncol=4)
            else:
                subplot2.legend(surfaceNames, loc='upper center', ncol=len(pltColors))
        subplot2.axis('off')

        subplotIndex += 1
        subplot3 = plt.subplot(subplotRow, subplotCol, subplotIndex)
        subplot3.imshow(images[b,].squeeze(), cmap='gray')
        for s in range(0, S):
            subplot3.plot(range(0, W), testOutputs[b, s, :].squeeze(), pltColors[s], linewidth=0.9)
        subplot3.axis('off')

        if not Output2Images:
            if MarkGTDisorder:
                plt.savefig(os.path.join(hps.outputDir, patientID_Index + "_MarkedImage_GT_Predict.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
            else:
                plt.savefig(os.path.join(hps.outputDir, patientID_Index + "_Image_GT_Predict.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(os.path.join(hps.outputDir, patientID_Index + "_GT_Predict.png"), dpi='figure',
                        bbox_inches='tight', pad_inches=0)
        plt.close()

    # check testOutputs whether violate surface-separation constraints
    testOutputs0 = testOutputs[:,0:-1,:]
    testOutputs1 = testOutputs[:, 1:, :]
    violateConstraintErrors = np.nonzero(testOutputs0 > testOutputs1)

    # final output result:
    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

    with open(os.path.join(hps.outputDir,f"output_{timeStr}.txt"), "w") as file:
        hps.printTo(file)
        file.write("\n=======net running parameters=========\n")
        file.write(f"B,S,H,W = {B, S, H, W}\n")
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



    print(f"============ End of Cross valiation test for OCT Multisurface Network: {hps.experimentName} ===========")


if __name__ == "__main__":
    main()