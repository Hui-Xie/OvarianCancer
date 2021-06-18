
# Cross Validation Test

import sys
import os
import yaml
import torch
import torch.nn as nn
from torch.utils import data


sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")


from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.SurfaceSegNet_Q import SurfaceSegNet_Q
from OCTData.OCTDataSet import  OCTDataSet
from OCTData.OCTDataUtilities import computeMASDError, batchPrediciton2OCTExplorerXML
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


        if hps.debug == True:
            print(f"sigma2.shape = {sigma2.shape}")
            print(f"mean of sigma2 = {torch.mean(sigma2, dim=[0,2])}")
            print(f"min of sigma2  = {torch.min(torch.min(sigma2, dim=0)[0], dim=-1)}")
            print(f"max of sigma2  = {torch.max(torch.max(sigma2, dim=0)[0], dim=-1)}")

        #output testID
        with open(os.path.join(hps.outputDir, f"testID.txt"), "w") as file:
            for id in testIDs:
                file.write(f"{id}\n")

        # Error Std and mean
        if hps.groundTruthInteger:
            testOutputs = (testOutputs + 0.5).int()  # as ground truth are integer, make the output also integers.

        # different applications need modify this.
        # get volumeIDs and volumeBscanStartIndexList
        volumeIDs = []
        volumeBscanStartIndexList = []
        B = len(testIDs)
        for i in range(0, B):  # we need consider the different Bscan numbers for different volumes.
            id = testIDs[i]
            if '_s000' == id[-5:]:
                volumeIDs.append(id[:, id.rfind("_s000")])
                volumeBscanStartIndexList.append(i)

        if hps.existGTLabel:
            stdSurfaceError, muSurfaceError, stdError, muError =  computeMASDError(testOutputs, testGts, volumeBscanStartIndexList, hPixelSize=hps.slicesPerPatient)
            testGts = testGts.cpu().numpy()

        images = images.cpu().numpy().squeeze()
        testOutputs = testOutputs.cpu().numpy()

        if outputXmlSegFiles:
            batchPrediciton2OCTExplorerXML(testOutputs, volumeIDs, volumeBscanStartIndexList, surfaceNames, hps.xmlOutputDir,
                                           refXMLFile=hps.refXMLFile,
                                           penetrationChar=hps.penetrationChar, penetrationPixels=hps.inputHeight, voxelSizeUnit=hps.voxelSizeUnit,
                                           voxelSizeX=hps.voxelSizeX, voxelSizeY=hps.voxelSizeY, voxelSizeZ=hps.voxelSizeZ)

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
            file.write(f"hausdorff Distance = {hausdorffD}\n")

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
        if ("OCT_Tongren" in hps.dataDir) or ("BES_3K" in hps.dataDir) :
            # example: "/home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
            # example: "/home/hxie1/data/OCT_Tongren/Glaucoma/209_OD_1554_Volume/30.jpg"  for glaucoma
            patientPath, filename = os.path.split(testIDs[b])
            patientID = os.path.basename(patientPath)
            if patientID not in patientIDList:
                patientIDList.append(patientID)
            volumePath = hps.imagesOutputDir+"/"+patientID
            if not os.path.exists(volumePath):
                os.makedirs(volumePath)  # recursive dir creation
            index = os.path.splitext(filename)[0]
            patientID_Index = patientID +"_" +index

        if "OCT_JHU" in hps.dataDir:
            # testIDs[0] = '/home/hxie1/data/OCT_JHU/preprocessedData/image/hc01_spectralis_macula_v1_s1_R_19.png'
            patientID_Index = os.path.splitext(os.path.basename(testIDs[b]))[0]  #e.g. hc01_spectralis_macula_v1_s1_R_19
            patient = patientID_Index[0:4] # e.g. hc01
            if "_s1_R_19" in patientID_Index and patient not in patientIDList:
                patientIDList.append(patient)

        if "OCT_Duke" in hps.dataDir:
            a = testIDs[b]
            patientID_Index = os.path.splitext(os.path.basename(a))[0]  # get file name stem, like "Control_1014_images_s50"


        if OutputNumImages ==0:
            continue

        f = plt.figure(frameon=False)
        # DPI = f.dpi
        DPI = 100.0

        if OutputNumImages==2:
            if ("OCT_Tongren" in hps.dataDir) or ("OCT_Duke" in hps.dataDir):
                subplotRow = 1
                subplotCol = 2
            else:
                subplotRow = 2
                subplotCol = 1
            imageFileName = patientID_Index + "_GT_Predict.png"

        elif OutputNumImages==1:
            subplotRow = 1
            subplotCol = 1
            imageFileName = patientID_Index + "_Predict.png"
        else:
            if W/H > 16/9:  # normal screen resolution rate is 16:9
                subplotRow = 3
                subplotCol = 1
            else:
                subplotRow = 1
                subplotCol = 3
            if OutputNumImages == 4:
                imageFileName = patientID_Index + f"_Raw_GT_Comparison_3S_center{comparisonSurfaceIndex:d}.png"
            else:
                imageFileName = patientID_Index + "_Raw_GT_Predict.png"
        f.set_size_inches(W * subplotCol / float(DPI), H * subplotRow / float(DPI))

        plt.margins(0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

        subplotIndex = 0

        if OutputNumImages>=3 and hps.existGTLabel:
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

        if OutputNumImages >=2 and hps.existGTLabel:
            subplotIndex += 1
            subplot2 = plt.subplot(subplotRow, subplotCol, subplotIndex)
            subplot2.imshow(images[b,].squeeze(), cmap='gray')
            for s in range(0, S):
                subplot2.plot(range(0, W), testGts[b, s, :].squeeze(), pltColors[s], linewidth=1.5)
            if needLegend:
                if ("OCT_Tongren" in hps.dataDir) or ("OCT_Duke" in hps.dataDir):
                    subplot2.legend(surfaceNames, loc='lower center', ncol=4, fontsize='x-small')
                else:
                    subplot2.legend(surfaceNames, loc='upper center', ncol=len(pltColors),  fontsize='x-small')
            subplot2.axis('off')

        subplotIndex += 1
        if 1 == subplotRow and 1 == subplotCol:
            subplot3 = plt
        else:
            subplot3 = plt.subplot(subplotRow, subplotCol, subplotIndex)
        subplot3.imshow(images[b,].squeeze(), cmap='gray')
        if OutputNumImages==4:
            ls = comparisonSurfaceIndex -1 # low index for comparison surface index
            if ls <0:
                ls =0
            hs =  comparisonSurfaceIndex +2 # high index
            if hs > S:
                hs = S
            GTColor = ['tab:blue', 'tab:brown', 'tab:olive']
            PredictionColor= ['tab:orange', 'tab:pink', 'tab:red',]
            legendList = []
            for s in range(ls, hs):
                subplot3.plot(range(0, W), testGts[b, s, :].squeeze(), GTColor[s%3], linewidth=1.5)
                legendList.append(f"GT_s{s}")
            for s in range(ls, hs):
                subplot3.plot(range(0, W), testOutputs[b, s, :].squeeze(), PredictionColor[s%3], linewidth=1.5)
                legendList.append(f"Prediction_s{s}")
            if needLegend:
                subplot3.legend(legendList, loc='lower center', ncol=2, fontsize='x-small')
        else:
            for s in range(0, S):
                subplot3.plot(range(0, W), testOutputs[b, s, :].squeeze(), pltColors[s], linewidth=1.5)
            if needLegend:
                if ("OCT_Tongren" in hps.dataDir) or ("OCT_Duke" in hps.dataDir):
                    subplot3.legend(surfaceNames, loc='lower center', ncol=4, fontsize='x-small')
                else:
                    subplot3.legend(surfaceNames, loc='upper center', ncol=len(pltColors), fontsize='x-small')
        subplot3.axis('off')

        if ("OCT_Tongren" in hps.dataDir) or ("BES_3K" in hps.dataDir) :
            plt.savefig(os.path.join(volumePath,imageFileName), dpi='figure', bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(os.path.join(hps.imagesOutputDir,imageFileName), dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"============ End of Cross valiation test for OCT Multisurface Network: {hps.experimentName} ===========")


if __name__ == "__main__":
    main()