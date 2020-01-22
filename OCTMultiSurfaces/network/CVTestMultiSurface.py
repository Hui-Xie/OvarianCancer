
# Cross Validation Test

import sys
import yaml

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


sys.path.append(".")
from OCTDataSet import OCTDataSet
from OCTUnet import OCTUnet
from OCTOptimization import *
from OCTTransform import *

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr

import matplotlib.pyplot as plt


def printUsage(argv):
    print("============ Cross Validation Test OCT MultiSurface Network =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def extractPaitentID(str):
    '''

       :param str: "/home/hxie1/data/OCT_Beijing/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
       :return: output: 4511_OD_29134_OCT06
       '''
    stem = str[:str.rfind("_Volume/")]
    patientID = stem[stem.rfind("/") + 1:]
    return patientID

def extractFileName(str):
    '''

    :param str: "/home/hxie1/data/OCT_Beijing/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
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

    # parse config file
    configFile = sys.argv[1]
    with open(configFile) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    experimentName = getStemName(configFile, removedSuffix=".yaml")
    print(f"Experiment: {experimentName}")

    dataDir = cfg["dataDir"]
    K = cfg["K_Folds"]
    k = cfg["fold_k"]
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

    testImagesPath = os.path.join(dataDir,"test", f"images_CV{k:d}.npy")
    testLabelsPath = os.path.join(dataDir,"test", f"surfaces_CV{k:d}.npy")
    testIDPath    = os.path.join(dataDir,"test", f"patientID_CV{k:d}.json")

    testData = OCTDataSet(testImagesPath, testLabelsPath, testIDPath, transform=None, device=device, sigma=sigma)

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

    logDir = dataDir + "/log/" + network + "/" + experimentName + "/testLog"
    if not os.path.exists(logDir):
        os.makedirs(logDir)  # recursive dir creation
    writer = SummaryWriter(log_dir=logDir)

    if outputDir=="":
        outputDir = dataDir + "/log/" + network + "/" + experimentName +"/testResult"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)  # recursive dir creation

    # test
    net.eval()
    with torch.no_grad():
        testBatch = 0
        for batchData in data.DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=0):
            testBatch += 1
            # S is surface location in (B,S,W) dimension, the predicted Mu
            S, _loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'])

            images = torch.cat((images, batchData['images'])) if testBatch != 1 else batchData['images'] # for output result
            testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
            testGts = torch.cat((testGts, batchData['GTs'])) if testBatch != 1 else batchData['GTs'] # Not Gaussian GTs
            testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for future output predict images

        # Error Std and mean
        stdSurfaceError, muSurfaceError,stdPatientError, muPatientError, stdError, muError = computeErrorStdMu(testOutputs, testGts,
                                                                                  slicesPerPatient=slicesPerPatient,
                                                                                  hPixelSize=hPixelSize)
    #generate predicted images
    B,S,W = testOutputs.shape
    images = images.cpu().numpy()
    testOutputs = testOutputs.cpu().numpy()
    testGts = testGts.cpu().numpy()
    patientIDList = []
    for b in range(B):
        # example: "/home/hxie1/data/OCT_Beijing/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
        patientID_Index = extractFileName(testIDs[b])  #e.g.: 4511_OD_29134_OCT06
        if "_OCT01" in patientID_Index:
            patientIDList.append(extractPaitentID(testIDs[b]))

        f = plt.figure(frameon=False)
        DPI = f.dpi
        H = 496
        f.set_size_inches(W*3/float(DPI), H/float(DPI))

        subplot1 = plt.subplot(1, 3, 1)
        subplot1.imshow(images[b,].squeeze(), cmap='gray')
        subplot1.axis('off')

        subplot2 = plt.subplot(1, 3, 2)
        subplot2.imshow(images[b,].squeeze(), cmap='gray')
        for s in range(0, S):
            subplot2.plot(range(0, W), testGts[b, s, :].squeeze(), linewidth=0.4)
        subplot2.axis('off')

        subplot3 = plt.subplot(1, 3, 3)
        subplot3.imshow(images[b,].squeeze(), cmap='gray')
        for s in range(0, S):
            subplot3.plot(range(0, W), testOutputs[b, s, :].squeeze(), linewidth=0.4)
        subplot3.axis('off')

        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.
        plt.savefig(os.path.join(outputDir, patientID_Index + "_Image_GT_Predict.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close()

    epoch = 0
    writer.add_scalar('ValidationError/mean(um)', muError, epoch)
    writer.add_scalar('ValidationError/stdDeviation(um)', stdError, epoch)
    writer.add_scalars('ValidationError/muSurface(um)', convertTensor2Dict(muSurfaceError), epoch)
    writer.add_scalars('ValidationError/muPatient(um)', convertTensor2Dict(muPatientError), epoch)
    writer.add_scalars('ValidationError/stdSurface(um)', convertTensor2Dict(stdSurfaceError), epoch)
    writer.add_scalars('ValidationError/stdPatient(um)', convertTensor2Dict(stdPatientError), epoch)

    with open(os.path.join(outputDir,"output.txt"), "w") as file:
        file.write(f"Test: {experimentName}\n")
        file.write(f"loadNetPath: {netPath}\n")
        file.write(f"B,S,W = {B,S,W}\n")
        file.write(f"stdSurfaceError = {stdSurfaceError}\n")
        file.write(f"muSurfaceError = {muSurfaceError}\n")
        file.write(f"patientIDList ={patientIDList}\n")
        file.write(f"stdPatientError = {stdPatientError}\n")
        file.write(f"muPatientError = {muPatientError}\n")
        file.write(f"stdError = {stdError}\n")
        file.write(f"muError = {muError}\n")


    print(f"============ End of Cross valiation test for OCT Multisurface Network: {experimentName} ===========")


if __name__ == "__main__":
    main()