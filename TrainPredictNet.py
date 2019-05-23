#  train predictive Network

import sys
import os
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

from LatentDataMgr import LatentDataMgr
from PredictModel import PredictModel
from NetMgr import NetMgr
from CustomizedLoss import FocalCELoss

# you may need to change the file name and log Notes below for every training.
trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/predictLog_20190522.txt'''
logNotes = r'''
Major program changes: 
                      the nunmber of filters in 1st layer in V model = 96
                      latent Vector size: 1536*51*49 (featureMap* slices * axisPlaneLatentVector)
                      PredictModel is convsDenseModule+FC network.
                      

            '''

logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')


def printUsage(argv):
    print("============Train Ovarian Cancer Predictive Model=============")
    print("Usage:")
    print(argv[0], "<netSavedPath> <fullPathOfTrainInputs>  <fullPathOfTestInputs> <fullPathOfLabels> ")


def main():
    if len(sys.argv) != 5:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    print(f'Program ID of Prdict Network training:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'Training log is in {trainLogFile}')
    print(f'.........')

    logging.info(f'Program ID of Prdict Network training:{os.getpid()}\n')
    logging.info(f'Program command: \n {sys.argv}')
    logging.info(logNotes)

    curTime = datetime.datetime.now()
    logging.info(f'\nProgram starting Time: {str(curTime)}')

    netPath = sys.argv[1]
    trainingInputsPath = sys.argv[2]
    testInputsPath = sys.argv[3]
    labelsPath = sys.argv[4]

    K = 2 # treatment response 1 or 0

    logging.info(f"Info: netPath = {netPath}\n")

    mergeTrainTestData = False

    trainDataMgr = LatentDataMgr(trainingInputsPath, labelsPath, logInfoFun=logging.info)

    if not mergeTrainTestData:
        testDataMgr = LatentDataMgr(testInputsPath, labelsPath, logInfoFun=logging.info)

    # ===========debug==================
    trainDataMgr.setOneSampleTraining(False)  # for debug
    if not mergeTrainTestData:
        testDataMgr.setOneSampleTraining(False)  # for debug
    useDataParallel = True  # for debug
    # ===========debug==================

    if mergeTrainTestData:
        trainDataMgr.expandInputsDir(testInputsPath, suffix="_Latent.npy")

    batchSize  = 8
    C = 1536  # number of input features
    H = 51    # height of input
    W = 49    # width of input

    trainDataMgr.setDataSize(batchSize, C, H, W, K,"TrainData")
                            # batchSize, depth, height, width, k, # do not consider lymph node with label 3
    if not mergeTrainTestData:
        testDataMgr.setDataSize(batchSize, C, H, W, K, "TestData")  # batchSize, depth, height, width, k
    net = PredictModel(C, K)

    trainDataMgr.setMixup(alpha=0.4, prob=0.5)  # set Mixup

    optimizer = optim.Adam(net.parameters())
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50, min_lr=1e-8)

    # Load network
    netMgr = NetMgr(net, netPath)
    bestTestDiceList = [0] * K
    if 2 == len(trainDataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")  # True for train
        logging.info(f'Program loads net from {netPath}.')
        bestTestPerf = netMgr.loadBestTestPerf(K)
        logging.info(f'Current best test dice: {bestTestDiceList}')
    else:
        logging.info(f"Network trains from scratch.")

    logging.info(net.getParametersScale())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ceWeight = torch.FloatTensor(trainDataMgr.getCEWeight()).to(device)
    focalLoss = FocalCELoss(weight=ceWeight)
    net.appendLossFunc(focalLoss, 1)

    net.to(device)
    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU > 1:
            logging.info(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net, device_ids=list(range(nGPU)), output_device=device)

    if useDataParallel:
        logging.info(net.module.lossFunctionsInfo())
    else:
        logging.info(net.lossFunctionsInfo())

    epochs = 150000
    logging.info(f"Hints: Optimal_Result = Yes = 1,  Optimal_Result = No = 0 ")

    logging.info(f"Epoch\tTrLoss\t" + "TrainAccuracy" + f"\t" + f"\tTsLoss\t" + f"TestAccuracy" )  # logging.info output head

    for epoch in range(epochs):
        # ================Training===============
        random.seed()
        trainingLoss = 0.0
        trainBatches = 0
        net.train()
        nTrainCorrect = 0
        nTrainTotal = 0
        trainAccuracy = 0
        if useDataParallel:
            lossWeightList = torch.Tensor(net.module.m_lossWeightList).to(device)

        for (inputs1, labels1Cpu), (inputs2, labels2Cpu) in zip(trainDataMgr.dataLabelGenerator(True),
                                                                trainDataMgr.dataLabelGenerator(True)):
            lambdaInBeta = trainDataMgr.getLambdaInBeta()
            inputs = inputs1 * lambdaInBeta + inputs2 * (1 - lambdaInBeta)
            inputs = torch.from_numpy(inputs).to(device, dtype=torch.float)
            labels1 = torch.from_numpy(labels1Cpu).to(device, dtype=torch.long)
            labels2 = torch.from_numpy(labels2Cpu).to(device, dtype=torch.long)

            if useDataParallel:
                optimizer.zero_grad()
                outputs = net.forward(inputs)
                loss = torch.tensor(0.0).cuda()
                for lossFunc, weight in zip(net.module.m_lossFuncList, lossWeightList):
                    if weight == 0:
                        continue
                    if lambdaInBeta != 0:
                        loss += lossFunc(outputs, labels1) * weight * lambdaInBeta
                    if 1 - lambdaInBeta != 0:
                        loss += lossFunc(outputs, labels2) * weight * (1 - lambdaInBeta)
                loss.backward()
                optimizer.step()
                batchLoss = loss.item()
            else:
                batchLoss = net.batchTrainMixup(inputs, labels1, labels2, lambdaInBeta)

            if lambdaInBeta == 1 :
               nTrainCorrect += labels1.eq(torch.argmax(outputs,dim=1)).sum()
               nTrainTotal += labels1.shape[0]

            trainingLoss += batchLoss
            trainBatches += 1

        if 0 != trainBatches:
            trainingLoss /= trainBatches
            trainAccuracy = nTrainCorrect/nTrainTotal


        # ================Test===============

        testLoss = 0.0
        testBatches = 0
        nTestCorrect = 0
        nTestTotal = 0
        testAccuracy = 0
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

                    nTestCorrect += labels1.eq(outputs).sum()
                    nTestTotal += labels1.shape[0]

                    testLoss += batchLoss
                    testBatches += 1


                # ===========print train and test progress===============
                if 0 != testBatches:
                    testLoss /= testBatches
                    testAccuracy = nTestCorrect/nTestTotal
                    lrScheduler.step(testLoss)
        else:
            lrScheduler.step(trainingLoss)

        logging.info(
            f'{epoch}\t{trainingLoss:.4f}\t' + f'{trainAccuracy:.3f}' + f'\t' +f'\t{testLoss:.4f}\t' + f'{testAccuracy:.3f}')

        # =============save net parameters==============
        if trainingLoss != float('inf') and trainingLoss != float('nan'):
            if mergeTrainTestData:
                netMgr.saveNet()
                if trainAccuracy > bestTestPerf:
                    bestTestPerf = trainAccuracy
                    netMgr.saveBest(bestTestPerf)

            else:
                netMgr.save(testAccuracy)
                if testAccuracy > bestTestPerf:  # compare the primary dice.
                    bestTestPerf = testAccuracy
                    netMgr.saveBest(bestTestPerf)
        else:
            logging.info(f"Error: training loss is infinity. Program exit.")
            sys.exit()

    torch.cuda.empty_cache()
    logging.info(f"=============END of Training of Ovarian Cancer Predict Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')


if __name__ == "__main__":
    main()
