# Ovarian cancer training program

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import random
import datetime

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
    print("============ TestTimeAugmentation Test Ovarian Cancer Response Network =============")
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
        trOptimalLoss = 0.0

        trPredictDict = {}
        trPredictProbDict = {}
        for batchData in data.DataLoader(trainData, batch_size=hps.batchSize, shuffle=True, num_workers=0):

            residualPredict, residualLoss, chemoPredict, chemoLoss, agePredict, ageLoss, survivalPredict, survivalLoss,optimalPredict, optimalLoss, predictProb\
                = net.forward(batchData['images'], GTs = batchData['GTs'])

            loss = hps.lossWeights[0]*residualLoss + hps.lossWeights[1]*chemoLoss + hps.lossWeights[2]*ageLoss \
                                                + hps.lossWeights[3]*survivalLoss + hps.lossWeights[4]*optimalLoss
            #if loss >= 1e-8:
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(hps.device))
            optimizer.step()

            trLoss += loss
            trResidualLoss += residualLoss
            trChemoLoss += chemoLoss
            trAgeLoss += ageLoss
            trSurvivalLoss += survivalLoss
            trOptimalLoss += optimalLoss
            trBatch += 1

            if hps.debug:
                for i in range(len(batchData['IDs'])):
                    MRN = batchData['IDs'][i]  # [0] is for list to string
                    trPredictDict[MRN] = {}
                    trPredictDict[MRN]['OptimalSurgery'] = -100  # todo: in the future
                    trPredictDict[MRN]['ResidualTumor'] = residualPredict[i].item()
                    trPredictDict[MRN]['ChemoResponse'] = chemoPredict[i].item()
                    trPredictDict[MRN]['Age'] = agePredict[i].item()
                    trPredictDict[MRN]['SurvivalMonths'] = survivalPredict[i].item()
                    trPredictDict[MRN]['OptimalResult'] = optimalPredict[i].item()

                    trPredictProbDict[MRN] = {}
                    trPredictProbDict[MRN]['Prob1'] = predictProb[i]
                    trPredictProbDict[MRN]['GT']    = batchData['GTs']['OptimalResult'][i]



            
        trLoss /=  trBatch
        trResidualLoss /= trBatch
        trChemoLoss /= trBatch
        trAgeLoss /= trBatch
        trSurvivalLoss /= trBatch
        trOptimalLoss /= trBatch

        if hps.debug:
            trGtDict = trainData.getGTDict()
            trResidualAcc = computeClassificationAccuracy(trGtDict, trPredictDict, 'ResidualTumor')
            trChemoAcc = computeClassificationAccuracy(trGtDict, trPredictDict, 'ChemoResponse')
            trAgeSqrtMSE = computeSqrtMSE(trGtDict, trPredictDict, 'Age')
            trSurvivalMonthsSqrtMSE = computeSqrtMSE(trGtDict, trPredictDict, 'SurvivalMonths')
            trOptimalAcc = computeClassificationAccuracy(trGtDict, trPredictDict, 'OptimalResult')

        if hps.debug and (epoch%100==0):
            curTime = datetime.datetime.now()
            timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
            if 8 == hps.colsGT:
                outputPredictDict2Csv8Cols(trPredictDict, hps.outputDir + f"/trainSetPredict_{timeStr}.csv")
            else:
                outputPredictDict2Csv6Cols(trPredictDict, hps.outputDir + f"/trainSetPredict_{timeStr}.csv")

            outputPredictProbDict2Csv(trPredictProbDict, hps.outputDir + f"/trainSetPredictProb_{timeStr}.csv")
            # print(f"Training prediction result has been output at {hps.outputDir}.")

        net.eval()
        predictDict= {}
        predictProbDict={}
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0
            validResidualLoss = 0.0
            validChemoLoss = 0.0
            validAgeLoss = 0.0
            validSurvivalLoss = 0.0
            validOptimalLoss = 0.0
            
            net.setStatus("validation")
            for batchData in data.DataLoader(validationData, batch_size=hps.batchSize, shuffle=True, num_workers=0):

                residualPredict, residualLoss, chemoPredict, chemoLoss, agePredict, ageLoss, survivalPredict, survivalLoss, optimalPredict, optimalLoss, predictProb\
                    = net.forward(batchData['images'], GTs=batchData['GTs'])

                loss = hps.lossWeights[0]*residualLoss + hps.lossWeights[1]*chemoLoss + hps.lossWeights[2]*ageLoss \
                        + hps.lossWeights[3]*survivalLoss  + hps.lossWeights[4]*optimalLoss
                # if loss >= 1e-8:
                validLoss += loss
                validResidualLoss += residualLoss
                validChemoLoss += chemoLoss
                validAgeLoss += ageLoss
                validSurvivalLoss += survivalLoss
                validOptimalLoss += optimalLoss
                validBatch += 1
                
                for i in range(len(batchData['IDs'])):
                    MRN = batchData['IDs'][i]  # [0] is for list to string
                    predictDict[MRN]={}
                    predictDict[MRN]['OptimalSurgery'] = -100  # todo: in the future
                    predictDict[MRN]['ResidualTumor'] = residualPredict[i].item()
                    predictDict[MRN]['ChemoResponse'] = chemoPredict[i].item()
                    predictDict[MRN]['Age'] = agePredict[i].item()
                    predictDict[MRN]['SurvivalMonths'] = survivalPredict[i].item()
                    predictDict[MRN]['OptimalResult'] = optimalPredict[i].item()

                    predictProbDict[MRN] = {}
                    predictProbDict[MRN]['Prob1'] = predictProb[i]
                    predictProbDict[MRN]['GT'] = batchData['GTs']['OptimalResult'][i]

            validLoss /= validBatch
            validResidualLoss /= validBatch
            validChemoLoss /= validBatch
            validAgeLoss /= validBatch
            validSurvivalLoss /= validBatch
            validOptimalLoss /= validBatch

        gtDict = validationData.getGTDict()
        residualAcc = computeClassificationAccuracy(gtDict,predictDict, 'ResidualTumor')
        chemoAcc = computeClassificationAccuracy(gtDict,predictDict, 'ChemoResponse')
        ageSqrtMSE = computeSqrtMSE(gtDict,predictDict, 'Age')
        survivalMonthsSqrtMSE = computeSqrtMSE(gtDict,predictDict, 'SurvivalMonths')
        optimalAcc = computeClassificationAccuracy(gtDict, predictDict, 'OptimalResult')

        lrScheduler.step(validLoss)
        # debug
        # print(f"epoch = {epoch}; trainLoss = {trLoss.item()};  validLoss = {validLoss.item()}")  # for smoke debug

        if hps.predictHeads[0]:
            writer.add_scalar('train/ResidualLoss', trResidualLoss, epoch)
            writer.add_scalar('validation/ResidualLoss', validResidualLoss, epoch)
            writer.add_scalar('ValidationAccuracy/ResiduaalTumorSizeAcc', residualAcc, epoch)
            writer.add_scalar('TrainingAccuracy/ResiduaalTumorSizeAcc', trResidualAcc, epoch) if hps.debug else None

        if hps.predictHeads[1]:
            writer.add_scalar('train/ChemoLoss', trChemoLoss, epoch)
            writer.add_scalar('validation/ChemoLoss', validChemoLoss, epoch)
            writer.add_scalar('ValidationAccuracy/ChemoResponseAcc', chemoAcc, epoch)
            writer.add_scalar('TrainingAccuracy/ChemoResponseAcc', trChemoAcc, epoch) if hps.debug else None

        if hps.predictHeads[2]:
            writer.add_scalar('train/AgeLoss', trAgeLoss, epoch)
            writer.add_scalar('validation/AgeLoss', validAgeLoss, epoch)
            writer.add_scalar('ValidationAccuracy/AgeSqrtMSE', ageSqrtMSE, epoch)
            writer.add_scalar('TrainingnAccuracy/AgeSqrtMSE', trAgeSqrtMSE, epoch) if hps.debug else None

        if hps.predictHeads[3]:
            writer.add_scalar('train/SurvivalLoss', trSurvivalLoss, epoch)
            writer.add_scalar('validation/SurvivalLoss', validSurvivalLoss, epoch)
            writer.add_scalar('ValidationAccuracy/SurvivalMonthsSqrtMSE', survivalMonthsSqrtMSE, epoch)
            writer.add_scalar('TrainingnAccuracy/SurvivalMonthsSqrtMSE', trSurvivalMonthsSqrtMSE,epoch) if hps.debug else None

        if hps.predictHeads[4]:
            writer.add_scalar('train/OptimalResultLoss', trOptimalLoss, epoch)
            writer.add_scalar('validation/OptimalResultLoss', validOptimalLoss, epoch)
            writer.add_scalar('ValidationAccuracy/OptimalResultAcc', optimalAcc, epoch)
            writer.add_scalar('TrainingnAccuracy/OptimalResultAcc', trOptimalAcc, epoch) if hps.debug else None

        writer.add_scalar('train/totalLoss', trLoss, epoch)
        writer.add_scalar('validation/totalLoss', validLoss, epoch)
        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        if validLoss < preValidLoss:
            net.updateRunParameter("validationLoss", validLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            net.updateRunParameter("ResiduaalTumorSizeAcc", residualAcc)
            net.updateRunParameter("ChemoResponseAcc", chemoAcc)
            net.updateRunParameter("AgeSqrtMSE", ageSqrtMSE)
            net.updateRunParameter("SurvivalMonthsSqrtMSE", survivalMonthsSqrtMSE)
            net.updateRunParameter("OptimalResultAcc", optimalAcc)
            preValidLoss = validLoss
            netMgr.saveNet(hps.netPath)

            curTime = datetime.datetime.now()
            timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
            if 8 == hps.colsGT:
                outputPredictDict2Csv8Cols(predictDict, hps.outputDir + f"/validationSetPredict_{timeStr}.csv")
            else:
                outputPredictDict2Csv6Cols(predictDict, hps.outputDir + f"/validationSetPredict_{timeStr}.csv")

            outputPredictProbDict2Csv(predictProbDict, hps.outputDir + f"/validationSetPredictProb_{timeStr}.csv")




    print("============ End of Training Ovarian Cancer  Network ===========")



if __name__ == "__main__":
    main()
