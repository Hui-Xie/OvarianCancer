# OCT to Systemic Disease training program

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import random
import datetime

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
from OCT2SysD_DataSet import OCT2SysD_DataSet
from OCT2SysD_Transform import OCT2SysD_Transform
from OCT2SysD_Net_A import OCT2SysD_Net_A
from OCT2SysD_Tools import *

sys.path.append("../..")
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.measure import  *


def printUsage(argv):
    print("============ Training of OCT to Systemic Disease Network =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")


def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")

    trainTransform = None
    if hps.augmentation:
        trainTransform = OCT2SysD_Transform(hps)
    # validationTransform = trainTransform
    # some people think validation supporting data augmentation benefits both learning rate decaying and generalization.

    trainData = OCT2SysD_DataSet("training", hps=hps, transform=trainTransform)
    validationData = OCT2SysD_DataSet("validation", hps=hps, transform=None) # only use data augmentation for training set

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # optimizer = optim.Adam(net.parameters(), lr=hps.learningRate, weight_decay=0)
    # refer to mobileNet v3 paper, use RMSprop optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=hps.learningRate, weight_decay=0, momentum=0.9)
    net.setOptimizer(optimizer)

    # lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-8, threshold=0.02, threshold_mode='rel')

    # math.log(0.5,0.98) = 34, this scheduler equals scale 0.5 per 100 epochs.
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.98, patience=3, min_lr=1e-8, threshold=0.02, threshold_mode='rel')
    net.setLrScheduler(lrScheduler)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("train")

    writer = SummaryWriter(log_dir=hps.logDir)

    # train
    epochs = 1360000
    preValidLoss = net.getRunParameter("validationLoss") if "validationLoss" in net.m_runParametersDict else 1e+8  # float 16 has maxvalue: 2048
    if net.training:
        initialEpoch = net.getRunParameter("epoch") if "epoch" in net.m_runParametersDict else 0
    else:
        initialEpoch = 0

    # application specific parameter
    hyptertensionPosWeight = torch.tensor(hps.hypertensionClassPercent[0] / hps.hypertensionClassPercent[1]).to(hps.device)
    appKey = 'hypertension_bp_plus_history$'

    for epoch in range(initialEpoch, epochs):
        random.seed()
        net.m_epoch = epoch
        net.setStatus("training")

        net.train()
        trBatch = 0
        trHyperTLoss = 0.0

        trPredictDict = {}
        trPredictProbDict = {}
        for batchData in data.DataLoader(trainData, batch_size=hps.batchSize, shuffle=True, num_workers=0):

            # merge B and S dimenions:
            B,S,C,H,W = batchData['images'].shape
            inputs = batchData['images'].view(B*S, C,H,W)

            x = net.forward(inputs)
            predict, predictProb, loss = net.computeBinaryLoss(x, GTs = batchData['GTs'], GTKey=appKey, posWeight=hyptertensionPosWeight)
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(hps.device))
            optimizer.step()

            trHyperTLoss += loss
            trBatch += 1

            if hps.debug:
                for i in range(len(batchData['IDs'])):
                    ID = batchData['IDs'][i]  # [0] is for list to string
                    trPredictDict[ID] = {}
                    trPredictDict[ID][appKey] = predict[i].item()

                    trPredictProbDict[ID] = {}
                    trPredictProbDict[ID]['Prob1'] = predictProb[i]
                    trPredictProbDict[ID]['GT']    = batchData['GTs'][appKey][i]

        trHyperTLoss /= trBatch

        if hps.debug:
            trGtDict = trainData.getGTDict()
            trHyperTAcc = computeClassificationAccuracy(trGtDict, trPredictDict, appKey)


        if hps.debug and (epoch%100==0):
            curTime = datetime.datetime.now()
            timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
            outputPredictProbDict2Csv(trPredictProbDict, hps.outputDir + f"/trainSetPredictProb_{timeStr}.csv")


        net.eval()
        predictDict= {}
        predictProbDict={}
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validHyperTLoss = 0.0

            net.setStatus("validation")
            for batchData in data.DataLoader(validationData, batch_size=hps.batchSize, shuffle=True, num_workers=0):

                # merge B and S dimenions:
                B, S, C, H, W = batchData['images'].shape
                inputs = batchData['images'].view(B * S, C, H, W)

                x = net.forward(inputs)
                predict, predictProb, loss = net.computeBinaryLoss(x, GTs=batchData['GTs'], GTKey=appKey, posWeight=hyptertensionPosWeight)

                validHyperTLoss +=loss
                validBatch += 1
                
                for i in range(len(batchData['IDs'])):
                    ID = batchData['IDs'][i]  # [0] is for list to string
                    predictDict[ID]={}
                    predictDict[ID][appKey] = predict[i].item()

                    predictProbDict[ID] = {}
                    predictProbDict[ID]['Prob1'] = predictProb[i]
                    predictProbDict[ID]['GT'] = batchData['GTs'][appKey][i]

            validHyperTLoss /= validBatch

        gtDict = validationData.getGTDict()
        hyperTAcc = computeClassificationAccuracy(gtDict,predictDict, appKey)


        lrScheduler.step(validHyperTLoss)
        # debug
        # print(f"epoch = {epoch}; trainLoss = {trLoss.item()};  validLoss = {validLoss.item()}")  # for smoke debug

        writer.add_scalar('train/HypertensionLoss', trHyperTLoss, epoch)
        writer.add_scalar('validation/HypertensionLoss', validHyperTLoss, epoch)
        writer.add_scalar('ValidationAccuracy/HypertensionAcc', hyperTAcc, epoch)
        writer.add_scalar('TrainingAccuracy/HypertensionAcc', trHyperTAcc, epoch) if hps.debug else None

        if validHyperTLoss < preValidLoss:
            net.updateRunParameter("validationLoss", validHyperTLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            preValidLoss = validHyperTLoss
            netMgr.saveNet(hps.netPath)

            curTime = datetime.datetime.now()
            timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
            outputPredictProbDict2Csv(predictProbDict, hps.outputDir + f"/validationSetPredictProb_{timeStr}.csv")




    print("============ End of Training OCT2SysDisease Network ===========")



if __name__ == "__main__":
    main()
