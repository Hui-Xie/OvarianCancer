
import datetime
import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
from FilesUtilities import *
from NetMgr import NetMgr

sys.path.append(".")
from OCDataSet import *
from VoteClassifier import *
from VoteBCEWithLogitsLoss import VoteBCEWithLogitsLoss


def printUsage(argv):
    print("============ Cross Validation Vote Classifier =============")
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
    mode = cfg["mode"]
    batchSize = cfg["batchSize"]
    latentDir = cfg["latentDir"]
    suffix = cfg["suffix"]
    patientResponsePath = cfg["patientResponsePath"]
    K_folds = cfg["K_folds"]
    fold_k = cfg["fold_k"]
    netPath = cfg["netPath"]+ "/" + experimentName
    rawF = cfg["rawF"]
    F = cfg["F"]
    device = eval(cfg["device"])  # convert string to class object.
    featureIndices = cfg["featureIndices"]

    # prepare data
    dataPartitions = OVDataPartition(latentDir, patientResponsePath, suffix, K_folds=K_folds, k=fold_k)
    positiveWeight = dataPartitions.getPositiveWeight().to(device)

    trainingData = OVDataSet('training', dataPartitions)
    validationData = OVDataSet('validation', dataPartitions)
    # testData = OVDataSet('test', dataPartitions)

    # construct network
    net = VoteClassifier()
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0)
    net.setOptimizer(optimizer)

    loss0 = VoteBCEWithLogitsLoss(pos_weight=positiveWeight, weightedVote=False)
    net.appendLossFunc(loss0, 0.5)
    loss1 = nn.BCEWithLogitsLoss(pos_weight=positiveWeight)
    net.appendLossFunc(loss1, 0.5)

    # Load network
    if 2 == len(getFilesList(netPath, ".pt")):
        netMgr = NetMgr(net, netPath, device)
        netMgr.loadNet("train")
        print(f"Fully Conneted Classifier load from  {netPath}")
        timeStr = getStemName(netPath)
    else:
        netMgr = NetMgr(net, netPath, device)
        print(f"Fully Conneted Classifier starts training from scratch, and save at {netPath}")

    logDir = latentDir + "/log/" + timeStr
    if not os.path.exists(logDir):
        os.makedirs(logDir)  # recursive dir creation
    writer = SummaryWriter(log_dir=logDir)


    # train
    epochs = 180000
    preLoss = 100000
    preAccuracy = 0

    for epoch in range(epochs):
        random.seed()
        net.m_epoch = epoch

        net.train()
        trBatch = 0
        trLoss = 0.0
        for inputs, labels, patientIDs in data.DataLoader(trainingData, batch_size=batchSize, shuffle=True, num_workers=0):
            trBatch +=1
            inputs = inputs.to(device, dtype=torch.float)
            gts = labels.to(device, dtype=torch.float)
            outputs, loss = net.forward(inputs, gts=gts)
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(device))
            optimizer.step()
            trLoss += loss
            trOutputs = torch.cat((trOutputs, outputs)) if trBatch != 1 else outputs
            trGts = torch.cat((trGts, gts)) if trBatch != 1 else gts

        trLoss = trLoss/trBatch
        trAccuracy = computeAccuracy(trOutputs, trGts)
        trTPR = computeTPR(trOutputs, trGts)
        trTNR = computeTNR(trOutputs, trGts)

        net.eval()
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0
            for inputs, labels, patientIDs in data.DataLoader(validationData, batch_size=batchSize, shuffle=False,num_workers=0):
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

        if validAccuracy > preAccuracy:
            preAccuracy = validAccuracy
            netMgr.saveNet(netPath)
            
     print("================End of Cross Validation==============")

if __name__ == "__main__":
    main()
