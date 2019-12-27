
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
from .OCTUnet import OCTUnet

from utilities.FilesUtilities import *
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
    network = cfg["network"]
    netPath = cfg["netPath"] + "/" + network + "/" + experimentName


    trainImagesPath = os.path.join(dataDir,"training", f"images_CV{k:d}.npy")
    trainLabelsPath  = os.path.join(dataDir,"training", f"surfaces_CV{k:d}.npy")
    trainIDPath     = os.path.join(dataDir,"training", f"patientID_CV{k:d}.json")

    validationImagesPath = os.path.join(dataDir,"validation", f"images_CV{k:d}.npy")
    validationLabelsPath = os.path.join(dataDir,"validation", f"surfaces_CV{k:d}.npy")
    validationIDPath    = os.path.join(dataDir,"validation", f"patientID_CV{k:d}.json")

    trainData = OCTDataSet(trainImagesPath, trainLabelsPath, trainIDPath, transform=None, device=device, sigma=sigma)
    validationData = OCTDataSet(validationImagesPath, validationLabelsPath, validationIDPath, transform=None, device=device, sigma=sigma)

    # construct network
    net = eval(network)(numSurfaces=numSurfaces, N=numStartFilters)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0)
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.4, patience=3000, min_lr=1e-8)


    loss = nn.KLDivLoss(reduction='batchmean').to(device)  # the input given is expected to contain log-probabilities
    net.appendLossFunc(loss, 1.0)

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
    epochs = 180000
    preLoss = 100000
    initialEpoch = 0  # for debug

    for epoch in range(initialEpoch, epochs):
        random.seed()
        net.m_epoch = epoch

        net.train()
        trBatch = 0
        trLoss = 0.0
        for inputs, labels, patientIDs in data.DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0):
            trBatch += 1
            inputs = inputs.to(device, dtype=torch.float)
            gts = labels.to(device, dtype=torch.float)
            outputs, loss = net.forward(inputs, gts=gts)
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(device))
            optimizer.step()
            trLoss += loss
            trOutputs = torch.cat((trOutputs, outputs)) if trBatch != 1 else outputs
            trGts = torch.cat((trGts, gts)) if trBatch != 1 else gts

        trLoss = trLoss / trBatch
        # print (f"epoch:{epoch}, trLoss = {trLoss.item()}")
        trAccuracy = computeAccuracy(trOutputs, trGts)
        trTPR = computeTPR(trOutputs, trGts)
        trTNR = computeTNR(trOutputs, trGts)

        lrScheduler.step(trLoss)

        net.eval()
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0
            for inputs, labels, patientIDs in data.DataLoader(validationData, batch_size=batchSize, shuffle=False,
                                                              num_workers=0):
                validBatch += 1
                inputs = inputs.to(device, dtype=torch.float)
                gts = labels.to(device, dtype=torch.float)
                outputs, loss = net.forward(inputs, gts=gts)
                validLoss += loss
                validOutputs = torch.cat((validOutputs, outputs)) if validBatch != 1 else outputs
                validGts = torch.cat((validGts, gts)) if validBatch != 1 else gts

            validLoss = validLoss / validBatch
            validAccuracy = computeAccuracy(validOutputs, validGts)
            validTPR = computeTPR(validOutputs, validGts)
            validTNR = computeTNR(validOutputs, validGts)

        writer.add_scalar('Loss/train', trLoss, epoch)
        writer.add_scalar('Loss/validation', validLoss, epoch)
        writer.add_scalar('TPR/train', trTPR, epoch)
        writer.add_scalar('TPR/validation', validTPR, epoch)
        writer.add_scalar('TNR/train', trTNR, epoch)
        writer.add_scalar('TNR/validation', validTNR, epoch)
        writer.add_scalar('Accuracy/train', trAccuracy, epoch)
        writer.add_scalar('Accuracy/validation', validAccuracy, epoch)
        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        if validAccuracy > preAccuracy and epoch > preTrainEpoch:
            preAccuracy = validAccuracy
            netMgr.saveNet(netPath)

    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()