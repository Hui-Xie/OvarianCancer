import sys
import os
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os


torchSummaryPath = "/home/hxie1/Projects/pytorch-summary/torchsummary"
sys.path.append(torchSummaryPath)
from torchsummary import summary

from SegDataMgr import SegDataMgr
from SegV3DModel import SegV3DModel
from SegV2DModel import SegV2DModel
from SegV2DModel_78 import SegV2DModel_78
from NetMgr  import NetMgr
from CustomizedLoss import FocalCELoss,BoundaryLoss

import numpy as np

# you may need to change the file name and log Notes below for every training.
trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/Log_20190523.txt'''
logNotes = r'''
Major program changes: 
                       merge train and test dataset;
                       for primary and metastases 3 classes classification
                       use conv-BN-Relu order;
                       use Dense module
                       Use ResPath
                       the nunmber of filters in 1st layer = 96
                        temperately test
                                             
                       
                       
            '''

logging.basicConfig(filename=trainLogFile,filemode='a+',level=logging.INFO, format='%(message)s')

def printUsage(argv):
    print("============Train Ovarian Cancer Segmentation V model=============")
    print("Usage:")
    print(argv[0], "<netSavedPath> <fullPathOfTrainImages>  <fullPathOfTrainLabels>  <2D|3D> <labelTuple>")
    print("eg. labelTuple: (0,1,2,3), or (0,1), (0,2)")

def main():
    if len(sys.argv) != 6:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    print(f'Program ID of Segmentaion V Model: {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'Training log is in {trainLogFile}')
    print(f'.........')

    logging.info(f'Program ID of Segmentaion V Model: {os.getpid()}\n')
    logging.info(f'Program command: \n {sys.argv}')
    logging.info(logNotes)

    curTime = datetime.datetime.now()
    logging.info(f'\nProgram starting Time: {str(curTime)}')



    netPath = sys.argv[1]
    imagesPath = sys.argv[2]
    labelsPath = sys.argv[3]
    is2DInput = True if sys.argv[4] == "2D" else False
    labelTuple = eval(sys.argv[5])
    K = len(labelTuple)

    logging.info(f"Info: netPath = {netPath}\n")

    mergeTrainTestData = True

    trainDataMgr = SegDataMgr(imagesPath, labelsPath, logInfoFun=logging.info)
    trainDataMgr.setRemainedLabel(3, labelTuple)

    if not mergeTrainTestData:
        testDataMgr = SegDataMgr(*trainDataMgr.getTestDirs(), logInfoFun=logging.info)
        testDataMgr.setRemainedLabel(3, labelTuple)

    # ===========debug==================
    restartTrainAfter100Epochs = True
    trainDataMgr.setOneSampleTraining(False)  # for debug
    if not mergeTrainTestData:
        testDataMgr.setOneSampleTraining(False)  # for debug
    useDataParallel = True  # for debug
    outputTrainDice = True
    if outputTrainDice:
        logging.info(f"Info: program output training dice.")
    else:
        logging.info(f"Info: program does not output training dice.")
    # ===========debug==================

    if mergeTrainTestData:
        trainDataMgr.expandInputsDir(trainDataMgr.getTestDirs()[0])
    trainDataMgr.buildSegSliceTupleList()


    if is2DInput:
        logging.info(f"Info: program uses 2D input.")
        trainDataMgr.setDataSize(8, 1, 281, 281, K, "TrainData")  # batchSize, depth, height, width, k, # do not consider lymph node with label 3
        if not mergeTrainTestData:
            testDataMgr.setDataSize(8, 1, 281, 281, K, "TestData")  # batchSize, depth, height, width, k
        net = SegV2DModel(96, K)
        # net = SegV2DModel_78(96, K)


    else:
        logging.info(f"Info: program uses 3D input.")
        trainDataMgr.setDataSize(4, 21, 281, 281, K, "TrainData")  # batchSize, depth, height, width, k
        if not mergeTrainTestData:
            testDataMgr.setDataSize(4, 21, 281, 281, K, "TestData")  # batchSize, depth, height, width, k
        net = SegV3DModel(K)

    trainDataMgr.setMaxShift(25, 0.5)             #translation data augmentation and its probability
    trainDataMgr.setFlipProb(0.3)                 #flip data augmentation
    trainDataMgr.setRot90sProb(0.3)               #rotate along 90, 180, 270
    # trainDataMgr.setJitterNoise(0.3, 1)           #add Jitter noise
    trainDataMgr.setAddedNoise(0.3, 0.0,  0.1)     #add gaussian noise augmentation after data normalization of [0,1]
    trainDataMgr.setMixup(alpha=0.4, prob=0.5)     # set Mixup

    optimizer = optim.Adam(net.parameters())
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-8)

    # Load network
    netMgr = NetMgr(net, netPath)
    bestTestDiceList = [0] * K
    if 2 == len(trainDataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")  # True for train
        logging.info(f'Program loads net from {netPath}.')
        bestTestDiceList = netMgr.loadBestTestPerf(K)
        logging.info(f'Current best test dice: {bestTestDiceList}')
    else:
        logging.info(f"Network trains from scratch.")

    logging.info(net.getParametersScale())
    logging.info(net.setDropoutProb(0.3))           # metastases is hard to learn, so it need a smaller dropout rate.

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ceWeight = torch.FloatTensor(trainDataMgr.getCEWeight()).to(device)
    focalLoss = FocalCELoss(weight=ceWeight)
    boundaryLoss = BoundaryLoss(lambdaCoeff=0.001, k=K, weight=ceWeight)
    if not restartTrainAfter100Epochs:
        net.appendLossFunc(focalLoss, 1)
        net.appendLossFunc(boundaryLoss, 0)
    else:
        net.appendLossFunc(focalLoss, 0.32)
        net.appendLossFunc(boundaryLoss, 0.68)

    fixedBoundaryLossWeight = True

    # logging.info model
    logging.info(f"\n====================Net Architecture===========================")
    stdoutBackup = sys.stdout
    with open(trainLogFile,'a+') as log:
        sys.stdout = log
        summary(net.cuda(), trainDataMgr.getInputSize())
    sys.stdout = stdoutBackup
    logging.info(f"===================End of Net Architecture =====================\n")

    net.to(device)
    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU >1:
            logging.info(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net, device_ids=[0,1,2,3], output_device=device)

    if useDataParallel:
        logging.info(net.module.lossFunctionsInfo())
    else:
        logging.info(net.lossFunctionsInfo())

    epochs = 15000
    logging.info(f"Hints: Test Dice_0 is the dice coeff for all non-zero labels")
    logging.info(f"Hints: Test Dice_1 is for primary cancer(green), \t\n test Dice_2 is for metastasis(yellow), \t\n and test Dice_3 is for invaded lymph node(brown).")
    logging.info(f"Hints: Test TPR_0 is the TPR for all non-zero labels")
    logging.info(f"Hints: Test TPR_1 is for primary cancer(green), \t\n TPR_2 is for metastasis(yellow), \t\n and TPR_3 is for invaded lymph node(brown).\n")
    diceHead1 = (f'Dice{i}' for i in labelTuple)  # generator object can be use only once.
    TPRHead1 = (f'TPR_{i}' for i in labelTuple)
    diceHead2 = (f'Dice{i}' for i in labelTuple)
    TPRHead2 = (f'TPR_{i}' for i in labelTuple)
    logging.info(f"Epoch\tTrLoss\t" + f"\t".join(diceHead1) + f"\t" + f"\t".join(TPRHead1)\
                    + f"\tTsLoss\t" + f"\t".join(diceHead2) + f"\t" + f"\t".join(TPRHead2))   # logging.info output head

    lastTrainingLoss = 1000
    for epoch in range(epochs):

        #================Update Loss weight==============
        lossWeightList = net.module.getLossWeightList() if useDataParallel else net.getLossWeightList()

        if fixedBoundaryLossWeight:
            if 105 == epoch:
                lossWeightList[0] = 0.32
                lossWeightList[1] = 0.68
                logging.info(f"before just epoch {epoch}, fix loss weight as {lossWeightList}")
                if useDataParallel:
                    net.module.updateLossWeightList(lossWeightList)
                else:
                    net.updateLossWeightList(lossWeightList)
        else:
            if len(lossWeightList) >1 and epoch > 100 and (epoch -100) % 5 == 0 :
                lossWeightList[0] -= 0.01
                lossWeightList[1] += 0.01
                if lossWeightList[0] < 0.01:
                    lossWeightList[0] = 0.01
                if lossWeightList[1] > 0.99:
                    lossWeightList[1] = 0.99

                if useDataParallel:
                    net.module.updateLossWeightList(lossWeightList)
                else:
                    net.updateLossWeightList(lossWeightList)


        #================Training===============
        random.seed()
        trainDiceSumList = [0 for _ in range(K)]
        trainDiceCountList = [0 for _ in range(K)]
        trainTPRSumList = [0 for _ in range(K)]
        trainTPRCountList = [0 for _ in range(K)]

        trainingLoss = 0.0
        trainBatches = 0
        net.train()
        if useDataParallel:
            lossWeightList = torch.Tensor(net.module.m_lossWeightList).to(device)

        for (inputs1, labels1Cpu), (inputs2, labels2Cpu) in zip(trainDataMgr.dataLabelGenerator(True), trainDataMgr.dataLabelGenerator(True)):
            lambdaInBeta = trainDataMgr.getLambdaInBeta()
            inputs = inputs1* lambdaInBeta + inputs2*(1-lambdaInBeta)
            inputs = torch.from_numpy(inputs).to(device, dtype=torch.float)
            labels1= torch.from_numpy(labels1Cpu).to(device, dtype=torch.long)
            labels2 = torch.from_numpy(labels2Cpu).to(device, dtype=torch.long)

            if useDataParallel:
                optimizer.zero_grad()
                outputs = net.forward(inputs)
                loss = torch.tensor(0.0).cuda()
                for lossFunc, weight in zip(net.module.m_lossFuncList, lossWeightList):
                    if weight == 0:
                        continue
                    if lambdaInBeta != 0:
                        loss += lossFunc(outputs, labels1) * weight*lambdaInBeta
                    if 1-lambdaInBeta != 0:
                        loss += lossFunc(outputs, labels2) * weight * (1-lambdaInBeta)
                loss.backward()
                optimizer.step()
                batchLoss = loss.item()
            else:
                batchLoss = net.batchTrainMixup(inputs, labels1, labels2, lambdaInBeta)

            if lambdaInBeta == 1 and outputTrainDice and epoch % 5 == 0:
                trainDiceSumList, trainDiceCountList, trainTPRSumList, trainTPRCountList \
                    = trainDataMgr.updateDiceTPRSumList(outputs, labels1Cpu, trainDiceSumList, trainDiceCountList, trainTPRSumList, trainTPRCountList)

            trainingLoss += batchLoss
            trainBatches += 1

        if 0 != trainBatches:
            trainingLoss /= trainBatches

        trainDiceAvgList = [x / (y + 1e-8) for x, y in zip(trainDiceSumList, trainDiceCountList)]
        trainTPRAvgList = [x / (y + 1e-8) for x, y in zip(trainTPRSumList, trainTPRCountList)]

        # ================Test===============
        testDiceSumList = [0 for _ in range(K)]
        testDiceCountList = [0 for _ in range(K)]
        testTPRSumList = [0 for _ in range(K)]
        testTPRCountList = [0 for _ in range(K)]
        testLoss = 0.0
        testBatches = 0
        if not mergeTrainTestData:
            net.eval()
            with torch.no_grad():
                for inputs, labelsCpu in testDataMgr.dataLabelGenerator(False):
                    inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labelsCpu)
                    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)  # return a copy

                    if useDataParallel:
                        outputs = net.forward(inputs)
                        loss = torch.tensor(0.0).cuda()
                        for lossFunc, weight in zip(net.module.m_lossFuncList, lossWeightList):
                            if weight == 0:
                                continue
                            loss += lossFunc(outputs, labels) * weight
                        batchLoss = loss.item()
                    else:
                        batchLoss, outputs = net.batchTest(inputs, labels)

                    testDiceSumList, testDiceCountList, testTPRSumList, testTPRCountList \
                        = testDataMgr.updateDiceTPRSumList(outputs, labelsCpu, testDiceSumList, testDiceCountList, testTPRSumList, testTPRCountList)

                    testLoss += batchLoss
                    testBatches += 1
                    #logging.info(f'batch={batches}: batchLoss = {batchLoss}')

                #===========print train and test progress===============
                if 0 != testBatches:
                    testLoss /= testBatches
                    lrScheduler.step(testLoss)
        else:
            lrScheduler.step(trainingLoss)

        testDiceAvgList = [x/(y+1e-8) for x,y in zip(testDiceSumList, testDiceCountList)]
        testTPRAvgList  = [x/(y+1e-8) for x, y in zip(testTPRSumList, testTPRCountList)]

        logging.info(f'{epoch}\t{trainingLoss:.4f}\t'+ f'\t'.join((f'{x:.3f}' for x in trainDiceAvgList))+f'\t'+ f'\t'.join( (f'{x:.3f}' for x in trainTPRAvgList))\
                              + f'\t{testLoss:.4f}\t'+ f'\t'.join((f'{x:.3f}' for x in testDiceAvgList))+ f'\t'+ f'\t'.join( (f'{x:.3f}' for x in testTPRAvgList)))

        # =============save net parameters==============
        if trainingLoss != float('inf') and trainingLoss != float('nan'):
            if mergeTrainTestData:
                netMgr.saveNet()
                if epoch % 5 == 0 and trainDiceAvgList[1] > bestTestDiceList[1]:
                    bestTestDiceList = trainDiceAvgList
                    netMgr.saveBest(bestTestDiceList)

            else:
                netMgr.save(testDiceAvgList)
                if testDiceAvgList[1] > 0.20  and testDiceAvgList[1] > bestTestDiceList[1]:  # compare the primary dice.
                    bestTestDiceList = testDiceAvgList
                    netMgr.saveBest(bestTestDiceList)
        else:
            logging.info(f"Error: training loss is infinity. Program exit.")
            sys.exit()

    torch.cuda.empty_cache()
    logging.info(f"=============END of Training of Ovarian Cancer Segmentation V Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')

if __name__ == "__main__":
    main()
