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
from OCT2SysD_Net import OCT2SysD_Net
from OCT2SysD_Tools import *

sys.path.append("../..")
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.measure import  *


def printUsage(argv):
    print("============ Training of OCT to Systemic Disease Network =============")
    print("=======input data is random single slice ===========================")
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
    if hps.trainAugmentation:
        trainTransform = OCT2SysD_Transform(hps)

    validationTransform = None
    if hps.validationAugmentation:
        validationTransform = OCT2SysD_Transform(hps)
    # some people think validation supporting data augmentation benefits both learning rate decaying and generalization.

    trainData = OCT2SysD_DataSet("training", hps=hps, transform=trainTransform)
    validationData = OCT2SysD_DataSet("validation", hps=hps, transform=validationTransform) # only use data augmentation for training set

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # optimizer = optim.Adam(net.parameters(), lr=hps.learningRate, weight_decay=0)

    # refer to mobileNet v3 paper, use RMSprop optimizer
    # optimizer = optim.RMSprop(net.parameters(), lr=hps.learningRate, weight_decay=0, momentum=0.9)

    # adaptive optimizers sometime are worse than SGD
    '''
    https://arxiv.org/abs/1705.08292
    The Marginal Value of Adaptive Gradient Methods in Machine Learning
    Adaptive optimization methods, which perform local optimization with a metric constructed from the history of iterates, 
    are becoming increasingly popular for training deep neural networks. Examples include AdaGrad, RMSProp, and Adam. 
    We show that for simple overparameterized problems, adaptive methods often find drastically different solutions 
    than gradient descent (GD) or stochastic gradient descent (SGD). We construct an illustrative binary classification 
    problem where the data is linearly separable, GD and SGD achieve zero test error, and AdaGrad, Adam, and RMSProp attain 
    test errors arbitrarily close to half. We additionally study the empirical generalization capability of adaptive methods
    on several state-of-the-art deep learning models. We observe that the solutions found by adaptive methods generalize 
    worse (often significantly worse) than SGD, even when these solutions have better training performance. 
    These results suggest that practitioners should reconsider the use of adaptive methods to train neural networks.
    '''
    optimizer = optim.SGD(net.parameters(), lr=hps.learningRate, weight_decay=hps.weightDecay, momentum=0)
    net.setOptimizer(optimizer)

    # lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-8, threshold=0.02, threshold_mode='rel')

    # math.log(0.5,0.98) = 34, this scheduler equals scale 0.5 per 100 epochs.
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=hps.lrSchedulerMode, factor=hps.lrDecayFactor, patience=hps.lrPatience, min_lr=1e-8, threshold=0.015, threshold_mode='rel')
    net.setLrScheduler(lrScheduler)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("train")

    writer = SummaryWriter(log_dir=hps.logDir)

    # train
    epochs = 1360000
    preValidLoss = net.getRunParameter("validationLoss") if "validationLoss" in net.m_runParametersDict else 1e+8  # float 16 has maxvalue: 2048
    preAccuracy = 0.5
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

            inputs = batchData['images']# B,3,H,W

            x = net.forward(inputs)
            predict, predictProb, loss = net.computeBinaryLoss(x, GTs = batchData['GTs'], GTKey=appKey, posWeight=hyptertensionPosWeight)
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(hps.device))
            optimizer.step()

            trHyperTLoss += loss
            trBatch += 1

            if hps.debug:
                for i in range(len(batchData['IDs'])):
                    ID = int(batchData['IDs'][i])  # [0] is for list to string
                    trPredictDict[ID] = {}
                    trPredictDict[ID][appKey] = predict[i].item()

                    trPredictProbDict[ID] = {}
                    trPredictProbDict[ID]['Prob1'] = predictProb[i]
                    trPredictProbDict[ID]['GT']    = batchData['GTs'][appKey][i]

            # debug
            # break

        trHyperTLoss /= trBatch

        if hps.debug:
            trGtDict = trainData.getGTDict()
            trHyperTAcc = computeClassificationAccuracy(trGtDict, trPredictDict, appKey)


        if hps.debug and (epoch%hps.debugOutputPeriod==0):
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
            batchSize = 1 if hps.TTA else hps.batchSize


            for batchData in data.DataLoader(validationData, batch_size=batchSize, shuffle=False, num_workers=0):

                # squeeze the extra dimension of data
                inputs = batchData['images'].squeeze(dim=0)
                batchData["GTs"]= {key: batchData["GTs"][key].squeeze(dim=0) for key in batchData["GTs"]}
                batchData['IDs'] = [v[0] for v in batchData['IDs']]  # erase tuple wrapper fot TTA

                x = net.forward(inputs)
                predict, predictProb, loss = net.computeBinaryLoss(x, GTs=batchData['GTs'], GTKey=appKey, posWeight=hyptertensionPosWeight)

                validHyperTLoss +=loss
                validBatch += 1

                B = predictProb.shape[0]
                for i in range(B):
                    ID = int(batchData['IDs'][i])  # [0] is for list to string
                    predictDict[ID]={}
                    predictDict[ID][appKey] = predict[i].item()

                    predictProbDict[ID] = {}
                    predictProbDict[ID]['Prob1'] = predictProb[i]
                    predictProbDict[ID]['GT'] = batchData['GTs'][appKey][i]

                # debug
                # break

            validHyperTLoss /= validBatch

        gtDict = validationData.getGTDict()
        hyperTAcc = computeClassificationAccuracy(gtDict,predictDict, appKey)
        Td_Acc_TPR_TNR_Sum = computeThresholdAccTPR_TNRSumFromProbDict(predictProbDict)


        if "min" == hps.lrSchedulerMode:
            lrScheduler.step(validHyperTLoss)
        else: # "max"
            lrScheduler.step(Td_Acc_TPR_TNR_Sum['Sum'])
        # debug
        # print(f"epoch = {epoch}; trainLoss = {trLoss.item()};  validLoss = {validLoss.item()}")  # for smoke debug

        writer.add_scalar('train/HypertensionLoss', trHyperTLoss, epoch)
        writer.add_scalar('validation/HypertensionLoss', validHyperTLoss, epoch)
        writer.add_scalar('ValidationAccuracy/HypertensionAcc', hyperTAcc, epoch)
        writer.add_scalars('ValidationAccuracy/threshold_ACC_TPR_TNR_Sum', Td_Acc_TPR_TNR_Sum, epoch)
        writer.add_scalar('TrainingAccuracy/HypertensionAcc', trHyperTAcc, epoch) if hps.debug else None
        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        #if validHyperTLoss < preValidLoss:
        # if  hyperTAcc > preAccuracy:
        if Td_Acc_TPR_TNR_Sum['Sum'] > preAccuracy:
            net.updateRunParameter("validationLoss", validHyperTLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            net.updateRunParameter("hyperTensionAccuracy", hyperTAcc)
            net.updateRunParameter("learningRate", optimizer.param_groups[0]['lr'])
            #preValidLoss = validHyperTLoss
            preAccuracy = Td_Acc_TPR_TNR_Sum['Sum']
            netMgr.saveNet(hps.netPath)

            curTime = datetime.datetime.now()
            timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
            outputPredictProbDict2Csv(predictProbDict, hps.outputDir + f"/validationSetPredictProb_{timeStr}.csv")
            # print("debug  ===")




    print("============ End of Training OCT2SysDisease Network ===========")



if __name__ == "__main__":
    main()
