# Ovarian cancer training program

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import random

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
from OVDataSet import OVDataSet
from OVDataTransform import OVDataTransform
from ResponseNet import ResponseNet
from OVTools import *

sys.path.append("../..")
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader


def printUsage(argv):
    print("============ Train Ovarian Cancer Response Network =============")
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

    trainTransform = OVDataTransform(hps)
    # validationTransform = trainTransform
    # validation supporting data augmentation benefits both learning rate decaying and generalization.

    trainData = OVDataSet("training", hps=hps, transform=trainTransform)
    validationData = OVDataSet("validation", hps=hps, transform=None) # only use data augmentation at training set

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    optimizer = optim.Adam(net.parameters(), lr=hps.learningRate, weight_decay=0)
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-8, threshold=0.02, threshold_mode='rel')
    net.setLrScheduler(lrScheduler)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("train")

    writer = SummaryWriter(log_dir=hps.logDir)

    # train
    epochs = 1360000
    preValidLoss = net.getRunParameter("validationLoss") if "validationLoss" in net.m_runParametersDict else 2041  # float 16 has maxvalue: 2048
    if net.training:
        initialEpoch = net.getRunParameter("epoch") if "epoch" in net.m_runParametersDict else 0
    else:
        initialEpoch = 0

    for epoch in range(initialEpoch, epochs):
        random.seed()
        net.m_epoch = epoch
        net.setStatus("training")

        net.train()
        trBatch = 0
        trLoss = 0.0
        trResidualLoss = 0.0
        trChemoLoss = 0.0
        trAgeLoss = 0.0
        trSurvivalLoss = 0.0
        
        for batchData in data.DataLoader(trainData, batch_size=hps.batchSize, shuffle=True, num_workers=0):
            trBatch += 1
            residualPredict, residualLoss, chemoPredict, chemoLoss, agePredict, ageLoss, survivalPredict, survivalLoss\
                = net.forward(batchData['images'].squeeze(dim=0), GTs = batchData['GTs'])

            loss = hps.lossWeights[0]*residualLoss + hps.lossWeights[1]*chemoLoss + hps.lossWeights[2]*ageLoss + hps.lossWeights[3]*survivalLoss
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(hps.device))
            optimizer.step()
            
            trLoss += loss
            trResidualLoss += residualLoss
            trChemoLoss += chemoLoss
            trAgeLoss += ageLoss
            trSurvivalLoss += survivalLoss
            
        trLoss /=  trBatch
        trResidualLoss /= trBatch
        trChemoLoss /= trBatch
        trAgeLoss /= trBatch
        trSurvivalLoss /= trBatch
        

        net.eval()
        predictDict= {}
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0
            validResidualLoss = 0.0
            validChemoLoss = 0.0
            validAgeLoss = 0.0
            validSurvivalLoss = 0.0
            
            net.setStatus("validation")
            for batchData in data.DataLoader(validationData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
                validBatch += 1
                residualPredict, residualLoss, chemoPredict, chemoLoss, agePredict, ageLoss, survivalPredict, survivalLoss\
                    = net.forward(batchData['images'].squeeze(dim=0), GTs=batchData['GTs'])

                validLoss += hps.lossWeights[0]*residualLoss + hps.lossWeights[1]*chemoLoss + hps.lossWeights[2]*ageLoss + hps.lossWeights[3]*survivalLoss
                validResidualLoss += residualLoss
                validChemoLoss += chemoLoss
                validAgeLoss += ageLoss
                validSurvivalLoss += survivalLoss
                
                MRN = batchData['IDs'][0]  # [0] is for list to string
                predictDict[MRN]={}
                predictDict[MRN]['ResidualTumor'] = residualPredict.item()
                predictDict[MRN]['ChemoResponse'] = chemoPredict.item()
                predictDict[MRN]['Age'] = agePredict.item()
                predictDict[MRN]['SurvivalMonths'] = survivalPredict.item()

            validLoss /= validBatch
            validResidualLoss /= validBatch
            validChemoLoss /= validBatch
            validAgeLoss /= validBatch
            validSurvivalLoss /= validBatch

        gtDict = validationData.getGTDict()
        residualAcc = computeClassificationAccuracy(gtDict,predictDict, 'ResidualTumor')
        chemoAcc = computeClassificationAccuracy(gtDict,predictDict, 'ChemoResponse')
        ageSqrtMSE = computeSqrtMSE(gtDict,predictDict, 'Age')
        survivalMonthsSqrtMSE = computeSqrtMSE(gtDict,predictDict, 'SurvivalMonths')


        lrScheduler.step(validLoss)
        # debug
        # print(f"epoch {epoch} ends...")  # for smoke debug

        writer.add_scalar('train/totalLoss', trLoss, epoch)
        writer.add_scalar('train/ResidualLoss', trResidualLoss, epoch)
        writer.add_scalar('train/ChemoLoss', trChemoLoss, epoch)
        writer.add_scalar('train/AgeLoss', trAgeLoss, epoch)
        writer.add_scalar('train/SurvivalLoss', trSurvivalLoss, epoch)

        writer.add_scalar('validation/totalLoss', validLoss, epoch)
        writer.add_scalar('validation/ResidualLoss', validResidualLoss, epoch)
        writer.add_scalar('validation/ChemoLoss', validChemoLoss, epoch)
        writer.add_scalar('validation/AgeLoss', validAgeLoss, epoch)
        writer.add_scalar('validation/SurvivalLoss', validSurvivalLoss, epoch)

        # error on resiudalSize, chemoResponse, Age, survival time
        writer.add_scalar('Accuracy/ResiduaalTumorSizeAcc', residualAcc, epoch)
        writer.add_scalar('Accuracy/ChemoResponseAcc', chemoAcc, epoch)
        writer.add_scalar('Accuracy/AgeSqrtMSE', ageSqrtMSE, epoch)
        writer.add_scalar('Accuracy/SurvivalMonthsSqrtMSE', survivalMonthsSqrtMSE, epoch)

        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        if validLoss < preValidLoss:
            net.updateRunParameter("validationLoss", validLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            net.updateRunParameter("ResiduaalTumorSizeAcc", residualAcc)
            net.updateRunParameter("ChemoResponseAcc", chemoAcc)
            net.updateRunParameter("AgeSqrtMSE", ageSqrtMSE)
            net.updateRunParameter("SurvivalMonthsSqrtMSE", survivalMonthsSqrtMSE)
            preValidLoss = validLoss
            netMgr.saveNet(hps.netPath)




    print("============ End of Training Ovarian Cancer  Network ===========")



if __name__ == "__main__":
    main()
