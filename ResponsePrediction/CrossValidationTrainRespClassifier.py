
#  training cross validation

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
from VoteClassifier import VoteClassifier
from FCClassifier import FCClassifier
from VoteBCEWithLogitsLoss import VoteBCEWithLogitsLoss
from AccuracyFunc import *


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
    batchSize = cfg["batchSize"]
    latentDir = cfg["latentDir"]
    suffix = cfg["suffix"]
    patientResponsePath = cfg["patientResponsePath"]
    K_folds = cfg["K_folds"]
    fold_k = cfg["fold_k"]
    network = cfg["network"]
    netPath = cfg["netPath"]+ "/"+network+ "/" + experimentName
    rawF = cfg["rawF"]
    F = cfg["F"]
    device = eval(cfg["device"])  # convert string to class object.
    featureIndices = cfg["featureIndices"]

    # prepare data
    dataPartitions = OVDataPartition(latentDir, patientResponsePath, suffix, K_folds=K_folds, k=fold_k)
    positiveWeight = dataPartitions.getPositiveWeight().to(device)

    trainingData = OVDataSet('training', dataPartitions, preLoadData=True)
    validationData = OVDataSet('validation', dataPartitions, preLoadData=True)
    # testData = OVDataSet('test', dataPartitions, preLoadData=True)

    # construct network
    net = eval(network)()
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0)
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.4, patience=3000, min_lr=1e-8)

    if isinstance(net, VoteClassifier):
        loss0 = VoteBCEWithLogitsLoss(pos_weight=positiveWeight, weightedVote=False)
        net.appendLossFunc(loss0, 0.5)
        loss1 = nn.BCEWithLogitsLoss(pos_weight=positiveWeight)
        net.appendLossFunc(loss1, 0.5)
        preTrainEpoch = 10000
    elif isinstance(net, FCClassifier):
        loss0 = nn.BCEWithLogitsLoss(pos_weight=positiveWeight)
        net.appendLossFunc(loss0, 1)
        preTrainEpoch = 0
    else:
        print(f"Error: maybe net error.")
        return

    # Load network
    if os.path.exists(netPath) and 2 == len(getFilesList(netPath, ".pt")):
        netMgr = NetMgr(net, netPath, device)
        netMgr.loadNet("train")
        print(f"Response Classifier load from  {netPath}")

    else:
        netMgr = NetMgr(net, netPath, device)
        print(f"Response Classifier starts training from scratch, and save at {netPath}")

    logDir = latentDir + "/log/"+ network+ "/" + experimentName
    if not os.path.exists(logDir):
        os.makedirs(logDir)  # recursive dir creation
    writer = SummaryWriter(log_dir=logDir)


    # train
    epochs = 180000
    preLoss = 100000
    preAccuracy = 0
    initialEpoch = 0  # for debug

    for epoch in range(initialEpoch, epochs):
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
        # print (f"epoch:{epoch}, trLoss = {trLoss.item()}")
        trAccuracy = computeAccuracy(trOutputs, trGts)
        trTPR = computeTPR(trOutputs, trGts)
        trTNR = computeTNR(trOutputs, trGts)

        lrScheduler.step(trLoss)

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
        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        if validAccuracy > preAccuracy and epoch > preTrainEpoch:
            preAccuracy = validAccuracy
            netMgr.saveNet(netPath)

    print("================End of Cross Validation==============")

if __name__ == "__main__":
    main()