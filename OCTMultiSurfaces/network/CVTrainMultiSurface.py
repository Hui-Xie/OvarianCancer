
import sys
import yaml
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
from OCTDataSet import OCTDataSet
from OCTUnet import OCTUnet
from measurement import *
from OCTTransform import *

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr


def printUsage(argv):
    print("============ Cross Validation Train OCT MultiSurface Network =============")
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
    lossFunc0 = cfg["lossFunc0"] # "nn.KLDivLoss(reduction='batchmean').to(device)"
    lossFunc0Epochs = cfg["lossFunc0Epochs"] #  the epoch number of using lossFunc0
    lossFunc1 = cfg["lossFunc1"] #  "nn.SmoothL1Loss().to(device)"
    lossFunc1Epochs = cfg["lossFunc1Epochs"] # the epoch number of using lossFunc1


    trainImagesPath = os.path.join(dataDir,"training", f"images_CV{k:d}.npy")
    trainLabelsPath  = os.path.join(dataDir,"training", f"surfaces_CV{k:d}.npy")
    trainIDPath     = os.path.join(dataDir,"training", f"patientID_CV{k:d}.json")

    validationImagesPath = os.path.join(dataDir,"validation", f"images_CV{k:d}.npy")
    validationLabelsPath = os.path.join(dataDir,"validation", f"surfaces_CV{k:d}.npy")
    validationIDPath    = os.path.join(dataDir,"validation", f"patientID_CV{k:d}.json")

    tainTransform = OCTDataTransform(prob=augmentProb, noiseStd=gaussianNoiseStd, saltPepperRate=saltPepperRate, saltRate=saltRate)

    trainData = OCTDataSet(trainImagesPath, trainLabelsPath, trainIDPath, transform=tainTransform, device=device, sigma=sigma)
    validationData = OCTDataSet(validationImagesPath, validationLabelsPath, validationIDPath, transform=None, device=device, sigma=sigma)

    # construct network
    net = eval(network)(numSurfaces=numSurfaces, N=numStartFilters)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0)
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.4, patience=250, min_lr=1e-8)

    # KLDivLoss is for Guassuian Ground truth for Unet
    loss0 = eval(lossFunc0) #nn.KLDivLoss(reduction='batchmean').to(device)  # the input given is expected to contain log-probabilities
    net.appendLossFunc(loss0, weight=1.0, epochs=lossFunc0Epochs)
    loss1 = eval(lossFunc1)
    net.appendLossFunc(loss1, weight=1.0, epochs=lossFunc1Epochs)


    # Load network
    if os.path.exists(netPath) and 2 == len(getFilesList(netPath, ".pt")):
        netMgr = NetMgr(net, netPath, device)
        netMgr.loadNet("train")
        print(f"Network load from  {netPath}")
    else:
        netMgr = NetMgr(net, netPath, device)
        print(f"Net starts training from scratch, and save at {netPath}")

    logDir = dataDir + "/log/" + network + "/" + experimentName
    if not os.path.exists(logDir):
        os.makedirs(logDir)  # recursive dir creation
    writer = SummaryWriter(log_dir=logDir)

    # train
    epochs = 1360000
    preLoss = 100000
    initialEpoch = 0  # for debug

    for epoch in range(initialEpoch, epochs):
        random.seed()
        net.m_epoch = epoch

        net.train()
        trBatch = 0
        trLoss = 0.0
        for batchData in data.DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0):
            trBatch += 1
            _softmaxOutputs, loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'])
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(device))
            optimizer.step()
            trLoss += loss

        trLoss = trLoss / trBatch
        lrScheduler.step(trLoss)

        net.eval()
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0
            for batchData in data.DataLoader(validationData, batch_size=batchSize, shuffle=False,
                                                              num_workers=0):
                validBatch += 1
                softmaxOutputs, loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'])
                validLoss += loss
                validOutputs = torch.cat((validOutputs, softmaxOutputs)) if validBatch != 1 else softmaxOutputs
                validGts = torch.cat((validGts, batchData['GTs'])) if validBatch != 1 else batchData['GTs'] # Not Gaussian GTs
                validIDs = validIDs + batchData['IDs'] if validBatch != 1 else batchData['IDs']  # for future output predict images

            validLoss = validLoss / validBatch

            # this needs modify
            predictMu, predictSigma2 = computeMuVariance(validOutputs)
            stdSurface, muSurface, stdPatient, muPatient, std, mu = computeErrorStdMu(predictMu, validGts,
                                                                                      slicesPerPatient=slicesPerPatient,
                                                                                      hPixelSize=hPixelSize)


        writer.add_scalar('Loss/train', trLoss, epoch)
        writer.add_scalar('Loss/validation', validLoss, epoch)
        writer.add_scalar('ValidationError/mean(um)', mu, epoch)
        writer.add_scalar('ValidationError/stdDeviation(um)', std, epoch)
        writer.add_scalars('ValidationError/muSurface(um)', convertTensor2Dict(muSurface), epoch)
        writer.add_scalars('ValidationError/muPatient(um)', convertTensor2Dict(muPatient), epoch)
        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        if validLoss < preLoss:
            preLoss = validLoss
            netMgr.saveNet(netPath)


    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()