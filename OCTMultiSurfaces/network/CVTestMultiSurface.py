
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


def printUsage(argv):
    print("============ Cross Validation Test OCT MultiSurface Network =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

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

    lossFunc0 = cfg["lossFunc0"] # "nn.KLDivLoss(reduction='batchmean').to(device)"
    lossFunc0Epochs = cfg["lossFunc0Epochs"] #  the epoch number of using lossFunc0
    lossFunc1 = cfg["lossFunc1"] #  "nn.SmoothL1Loss().to(device)"
    lossFunc1Epochs = cfg["lossFunc1Epochs"] # the epoch number of using lossFunc1

    # Proximal IPM Optimization
    useProxialIPM = cfg['useProxialIPM']
    learningStepIPM =cfg['learningStepIPM'] # 0.1
    nIterationIPM =cfg['nIterationIPM'] # : 50

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

    # according config file to update config parameter, maybe not necessary
    net.updateConfigParameter('useProxialIPM', useProxialIPM)
    net.updateConfigParameter("learningStepIPM", learningStepIPM)
    net.updateConfigParameter("nIterationIPM", nIterationIPM)

    outputDir = dataDir + "/log/" + network + "/" + experimentName +"/testResult"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)  # recursive dir creation
    writer = SummaryWriter(log_dir=outputDir)

    # test
    net.eval()
    with torch.no_grad():
        testBatch = 0
        for batchData in data.DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=0):
            testBatch += 1
            # S is surface location in (B,S,W) dimension, the predicted Mu
            S, _loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'])
            testOutputs = torch.cat((testOutputs, S)) if testBatch != 1 else S
            testGts = torch.cat((testGts, batchData['GTs'])) if testBatch != 1 else batchData['GTs'] # Not Gaussian GTs
            testIDs = testIDs + batchData['IDs'] if testBatch != 1 else batchData['IDs']  # for future output predict images

        # Error Std and mean
        stdSurfaceError, muSurfaceError,stdPatientError, muPatientError, stdError, muError = computeErrorStdMu(testOutputs, testGts,
                                                                                  slicesPerPatient=slicesPerPatient,
                                                                                  hPixelSize=hPixelSize)
        epoch = 0
        writer.add_scalar('ValidationError/mean(um)', muError, epoch)
        writer.add_scalar('ValidationError/stdDeviation(um)', stdError, epoch)
        writer.add_scalars('ValidationError/muSurface(um)', convertTensor2Dict(muSurfaceError), epoch)
        writer.add_scalars('ValidationError/muPatient(um)', convertTensor2Dict(muPatientError), epoch)
        writer.add_scalars('ValidationError/stdSurface(um)', convertTensor2Dict(stdSurfaceError), epoch)
        writer.add_scalars('ValidationError/stdPatient(um)', convertTensor2Dict(stdPatientError), epoch)


        print(f"Test: {experimentName}")
        print(f"loadNetPath: {netPath}")
        print(f"stdSurfaceError = {stdSurfaceError}")
        print(f"muSurfaceError = {muSurfaceError}")
        print(f"stdPatientError = {stdPatientError}")
        print(f"muPatientError = {muPatientError}")
        print(f"stdError = {stdError}")
        print(f"muError = {muError}")


    print("============ End of Cross valiation test for OCT Multisurface Network ===========")


if __name__ == "__main__":
    main()