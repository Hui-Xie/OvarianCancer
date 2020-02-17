
# Cross Validation Test

import sys
import yaml

import torch
import torch.nn as nn
from torch.utils import data


sys.path.append(".")
from OCTDataSet import *
from IVUSUnet import IVUSUnet
from OCTOptimization import *
from OCTTransform import *
from OCTAugmentation import *

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr

import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from numpy import genfromtxt

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
    MarkPredictDisorder = True
    OutputPredictImages = True

    # parse config file
    configFile = sys.argv[1]
    with open(configFile) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    experimentName = getStemName(configFile, removedSuffix=".yaml")
    print(f"Experiment: {experimentName}")

    assert "IVUS" in experimentName

    dataDir = cfg["dataDir"]
    K = cfg["K_Folds"]
    k = cfg["fold_k"]

    groundTruthInteger = cfg["groundTruthInteger"]
    numSurfaces = cfg["numSurfaces"]
    sigma = cfg["sigma"]  # for gausssian ground truth
    device = eval(cfg["device"])  # convert string to class object.
    batchSize = cfg["batchSize"]
    numStartFilters = cfg["startFilters"]  # the num of filter in first layer of Unet

    slicesPerPatient = cfg["slicesPerPatient"] # 31
    hPixelSize = cfg["hPixelSize"] #  3.870  # unit: micrometer, in y/height direction

    augmentProb = cfg["augmentProb"]
    gaussianNoiseStd = cfg["gaussianNoiseStd"]  # gausssian nosie std with mean =0
    # for salt-pepper noise
    saltPepperRate= cfg["saltPepperRate"]   # rate = (salt+pepper)/allPixels
    saltRate= cfg["saltRate"]  # saltRate = salt/(salt+pepper)
    lacingWidth = cfg["lacingWidth"]
    if "IVUS" in dataDir:
        rotation = cfg["rotation"]
    else:
        rotation = False
    TTA = cfg["TTA"]  # Test-Time Augmentation
    TTA_StepDegree = cfg["TTA_StepDegree"]

    network = cfg["network"]
    netPath = cfg["netPath"] + "/" + network + "/" + experimentName
    loadNetPath = cfg['loadNetPath']
    if "" != loadNetPath:
        netPath = loadNetPath
    outputDir = cfg["outputDir"]

    lossFunc0 = cfg["lossFunc0"] # "nn.KLDivLoss(reduction='batchmean').to(device)"
    lossFunc0Epochs = cfg["lossFunc0Epochs"] #  the epoch number of using lossFunc0
    lossFunc1 = cfg["lossFunc1"] #  "nn.SmoothL1Loss().to(device)"
    lossFunc1Epochs = cfg["lossFunc1Epochs"] # the epoch number of using lossFunc1

    # Proximal IPM Optimization
    useProxialIPM = cfg['useProxialIPM']
    if useProxialIPM:
        learningStepIPM = cfg['learningStepIPM']  # 0.1
        maxIterationIPM = cfg['maxIterationIPM']  # : 50
        criterionIPM = cfg['criterionIPM']

    useDynamicProgramming = cfg['useDynamicProgramming']
    usePrimalDualIPM = cfg['usePrimalDualIPM']
    useCEReplaceKLDiv = cfg['useCEReplaceKLDiv']

    if -1==k and 0==K:  # do not use cross validation
        testImagesPath = os.path.join(dataDir, "test", f"images.npy")
        testLabelsPath = os.path.join(dataDir, "test", f"surfaces.npy")
        testIDPath = os.path.join(dataDir, "test", f"patientID.json")
    else:  # use cross validation
        testImagesPath = os.path.join(dataDir,"test", f"images_CV{k:d}.npy")
        testLabelsPath = os.path.join(dataDir,"test", f"surfaces_CV{k:d}.npy")
        testIDPath    = os.path.join(dataDir,"test", f"patientID_CV{k:d}.json")



    # construct network
    net = eval(network)(numSurfaces=numSurfaces, N=numStartFilters)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=device)

    # KLDivLoss is for Guassuian Ground truth for Unet
    loss0 = eval(lossFunc0) #nn.KLDivLoss(reduction='batchmean').to(device)  # the input given is expected to contain log-probabilities
    net.appendLossFunc(loss0, weight=1.0, epochs=lossFunc0Epochs)
    loss1 = eval(lossFunc1)
    net.appendLossFunc(loss1, weight=1.0, epochs=lossFunc1Epochs)

    # Load network
    if os.path.exists(netPath) and len(getFilesList(netPath, ".pt")) >= 2 :
        netMgr = NetMgr(net, netPath, device)
        netMgr.loadNet("test")
        print(f"Network load from  {netPath}")
    else:
        print(f"Can not find pretrained network for test!")

    # according config file to update config parameter
    net.updateConfigParameter('useProxialIPM', useProxialIPM)
    if useProxialIPM:
        net.updateConfigParameter("learningStepIPM", learningStepIPM)
        net.updateConfigParameter("maxIterationIPM", maxIterationIPM)
        net.updateConfigParameter("criterion", criterionIPM)

    net.updateConfigParameter("useDynamicProgramming", useDynamicProgramming)
    net.updateConfigParameter("usePrimalDualIPM", usePrimalDualIPM)
    net.updateConfigParameter("useCEReplaceKLDiv", useCEReplaceKLDiv)

    if outputDir=="":
        outputDir = dataDir + "/log/" + network + "/" + experimentName +"/testResult"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)  # recursive dir creation

    # test
    net.eval()
    with torch.no_grad():

        # Test-Time Augmentation
        nCountTTA = 0
        for TTADegree in range(0, 360, TTA_StepDegree):
            nCountTTA += 1
            testData = OCTDataSet(testImagesPath, testLabelsPath, testIDPath, transform=None, device=device, sigma=sigma,
                                  lacingWidth=lacingWidth, TTA=TTA, TTA_Degree=TTADegree)
            testBatch = 0
            for batchData in data.DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=0):
                testBatch += 1
                # S is surface location in (B,S,W) dimension, the predicted Mu
                S, _loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'])

                images = torch.cat((images, batchData['images'])) if testBatch != 1 else batchData['images'] # for output result
                testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
                testGts = torch.cat((testGts, batchData['GTs'])) if testBatch != 1 else batchData['GTs'] # Not Gaussian GTs
                testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for output predict images' ID

            # Delace polar images and labels
            images.squeeze_(dim=1)  # squeeze channel dim
            if 0 != lacingWidth:
                images, testOutputs = delacePolarImageLabel(images, testOutputs, lacingWidth)
                testGts = delacePolarLabel(testGts, lacingWidth)

            if 0 != TTADegree: # retate back
                images,testOutputs = polarImageLabelRotate_Tensor(images, testOutputs, -TTADegree)
                testGts = polarLabelRotate_Tensor(testGts, -TTADegree)

            testOutputsTTA = testOutputsTTA + testOutputs if 1 != nCountTTA else testOutputs

            if (TTA ==False or 0 == TTA_StepDegree):
                break

        testOutputs = testOutputsTTA/nCountTTA  # average to get final prediction value

    # Error Std and mean
    if groundTruthInteger:
        testOutputs = (testOutputs + 0.5).int()  # as ground truth are integer, make the output also integers.
    stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(testOutputs, testGts,
                                                                                             slicesPerPatient=slicesPerPatient,
                                                                                             hPixelSize=hPixelSize)
    #generate predicted images
    images = images.cpu().numpy().squeeze()
    B,H,W = images.shape
    B, S, W = testOutputs.shape
    testOutputs = testOutputs.cpu().numpy()
    testGts = testGts.cpu().numpy()
    patientIDList = []

    outputTxtDir = os.path.join(outputDir, "text")
    outputImageDir = os.path.join(outputDir, "images")
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
        np.savetxt(lumenPredictFile, cartesianLabel[0,], fmt="%.1f", delimiter=", ")
        np.savetxt(mediaPredictFile, cartesianLabel[1,], fmt="%.1f", delimiter=", ")


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
            f.set_size_inches(W/ float(DPI), H*3 / float(DPI))
            subplotRow = 3
            subplotCol = 1
        else:
            f.set_size_inches(W*3/float(DPI), H/float(DPI))
            subplotRow = 1
            subplotCol = 3

        plt.margins(0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

        subplot1 = plt.subplot(subplotRow, subplotCol, 1)
        subplot1.imshow(imageb, cmap='gray')
        subplot1.axis('off')

        subplot2 = plt.subplot(subplotRow, subplotCol, 2)
        subplot2.imshow(imageb, cmap='gray')
        subplot2.plot(lumenGTLabel[:, 0], lumenGTLabel[:, 1], linewidth=0.4)
        subplot2.plot(mediaGTLabel[:, 0], mediaGTLabel[:, 1], linewidth=0.4)
        subplot2.axis('off')

        subplot3 = plt.subplot(subplotRow, subplotCol, 3)
        subplot3.imshow(imageb, cmap='gray')
        for s in range(0, S):
            subplot3.plot(cartesianLabel[s,:,0], cartesianLabel[s,:,1], linewidth=0.4)
        subplot3.axis('off')


        plt.savefig(os.path.join(outputImageDir, patientID + "_Image_GT_Predict.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close()

    # check testOutputs whehter violate surface-separation constraints
    testOutputs0 = testOutputs[:,0:-1,:]
    testOutputs1 = testOutputs[:, 1:, :]
    violateConstraintErrors = np.nonzero(testOutputs0 > testOutputs1)

    # final output result:
    with open(os.path.join(outputDir,"output.txt"), "w") as file:
        file.write(f"Test: {experimentName}\n")
        file.write(f"loadNetPath: {netPath}\n")
        file.write(f"config file: {configFile}")
        file.write(f"B,S,H,W = {B,S,H, W}\n")
        file.write(f"net.m_configParametersDict:\n")
        [file.write(f"\t{key}:{value}\n") for key, value in net.m_configParametersDict.items()]
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



    print(f"============ End of Test: {experimentName} ===========")


if __name__ == "__main__":
    main()