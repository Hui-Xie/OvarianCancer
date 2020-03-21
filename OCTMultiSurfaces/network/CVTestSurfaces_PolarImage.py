
# Cross Validation Test, and support automatic layer size

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
from OCTAugmentation import *

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from numpy import genfromtxt
import datetime

# import matlab.engine

sys.path.append("../dataPrepare_IVUS")
from PolarCartesianConverter import PolarCartesianConverter

testGTSegDir = "/home/hxie1/data/IVUS/Test_Set/Data_set_B/LABELS_obs1"


def printUsage(argv):
    print("============ Cross Validation Test OCT MultiSurface Network =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def extractPaitentID(str):  # for Tongren data
    '''

       :param str: "/home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
       :return: output: 4511_OD_29134_OCT06
       '''
    stem = str[:str.rfind("_Volume/")]
    patientID = stem[stem.rfind("/") + 1:]
    return patientID

def extractFileName(str): # for Tongren data
    '''

    :param str: "/home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
    :return: output: 4511_OD_29134_OCT06
    '''
    stem = str[:str.rfind("_Volume/")]
    patientID = stem[stem.rfind("/")+1:]
    OCTIndex = str[str.rfind("_"):-4]
    return patientID+OCTIndex

def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # debug:
    MarkGTDisorder = False
    MarkPredictDisorder = False
    OutputPredictImages = False
    Output2Images = True

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



    # construct network
    net = eval(hps.network)(hps.inputHight, hps.inputWidth, inputChannels=hps.inputChannels, nLayers=hps.nLayers, numSurfaces=hps.numSurfaces, N=hps.numStartFilters)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # KLDivLoss is for Guassuian Ground truth for Unet
    loss0 = eval(hps.lossFunc0) #nn.KLDivLoss(reduction='batchmean').to(device)  # the input given is expected to contain log-probabilities
    net.appendLossFunc(loss0, weight=1.0, epochs=hps.lossFunc0Epochs)
    loss1 = eval(hps.lossFunc1)
    net.appendLossFunc(loss1, weight=1.0, epochs=hps.lossFunc1Epochs)

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

        # Test-Time Augmentation
        nCountTTA = 0
        for TTADegree in range(0, 360, hps.TTA_StepDegree):
            nCountTTA += 1
            testData = OCTDataSet(testImagesPath, testLabelsPath, testIDPath, transform=None, device=hps.device, sigma=hps.sigma,
                                  lacingWidth=hps.lacingWidth, TTA=hps.TTA, TTA_Degree=hps.TTADegree, scaleNumerator=hps.scaleNumerator,
                                  scaleDenominator=hps.scaleDenominator, gradChannels=hps.gradChannels)
            testBatch = 0
            for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
                testBatch += 1
                # S is surface location in (B,S,W) dimension, the predicted Mu
                S, _loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'])

                images = torch.cat((images, batchData['images'])) if testBatch != 1 else batchData['images'] # for output result
                testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
                testGts = torch.cat((testGts, batchData['GTs'])) if testBatch != 1 else batchData['GTs'] # Not Gaussian GTs
                testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for output predict images' ID

            images = images[:,0,:,:]  # erase grad channels
            images.squeeze_(dim=1)  # squeeze channel dim

            # scale back: # this will change the Height of polar image
            if 1 != hps.scaleNumerator or 1 != hps.scaleDenominator:
                images = scalePolarImage(images, hps.scaleDenominator, hps.scaleNumerator)
                testOutputs = scalePolarLabel(testOutputs, hps.scaleDenominator, hps.scaleNumerator)
                testGts = scalePolarLabel(testGts, hps.scaleDenominator, hps.scaleNumerator)

            # Delace polar images and labels
            if 0 != hps.lacingWidth:
                images, testOutputs = delacePolarImageLabel(images, testOutputs, hps.lacingWidth)
                testGts = delacePolarLabel(testGts, hps.lacingWidth)

            if 0 != TTADegree: # retate back
                images,testOutputs = polarImageLabelRotate_Tensor(images, testOutputs, -TTADegree)
                testGts = polarLabelRotate_Tensor(testGts, -TTADegree)

            testOutputsTTA = testOutputsTTA + testOutputs if 1 != nCountTTA else testOutputs

            if (hps.TTA ==False or 0 == hps.TTA_StepDegree):
                break

            break

        testOutputs = testOutputsTTA/nCountTTA  # average to get final prediction value

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
        DPI = f.dpi
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