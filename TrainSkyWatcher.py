#  train Skywatcher Model for segmentation and treatment response together

import sys
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

from Image3dResponseDataMgr import Image3dResponseDataMgr
from SkyWatcherModel import SkyWatcherModel
from NetMgr import NetMgr
from CustomizedLoss import FocalCELoss, BoundaryLoss

# you may need to change the file name and log Notes below for every training.
trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/log_SkyWatcher_20190605.txt'''
logNotes = r'''
Major program changes: 
                      

Experiment setting for Image3d ROI to response:
Input CT data: 147*281*281  3D CT raw image ROI
segmentation label: 127*255*255 segmentation label with value (0,1,2) which erases lymph node label

This is a multi-task learning. 

Predictive Model: 1,  first 3-layer dense conv block with channel size 24.
                  2,  and 6 dense conv DownBB blocks,  each of which includes a stride 2 conv and 3-layers dense conv block; 
                  3,  and 3 fully connected layers  changes the tensor into size 2*1;
                  4,  final a softmax for binary classification;
                  Total network learning parameters are 236K.
                  Network architecture is referred at https://github.com/Hui-Xie/OvarianCancer/blob/master/Image3dPredictModel.py

response Loss Function:   Cross Entropy with weight [3.3, 1.4] for [0,1] class separately, as [0,1] uneven distribution.
segmenation loss function: focus loss + boundary loss

Data:   training data has 130 patients, and test data has 32 patients with training/test rate 80/20.
        We used patient ID as index to order all patients data, and then used about the first 80% of patients as training data, 
        and the remaining 20% of patients as test data. 
        Sorting with patient ID is to make sure the division of training and test set is blind to the patient's detailed stage, 
        shape and size of cancer.  
        Therefore you will see that patient IDs of all test data are beginning at 8 or 9. 
        This training/test division is exactly same with segmentation network experiment before. 

Training strategy:  50% probability of data are mixed up with beta distribution with alpha =0.4, to feed into network for training. 
                    No other data augmentation, and no dropout.  

                    Learning Scheduler:  Reduce learning rate on  plateau, and learning rate patience is 30 epochs.                                

            '''

logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')


def printUsage(argv):
    print("============Train SkyWatcher Model for Ovarian Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <fullPathOfTrainInputs>  <fullPathOfTestInputs> <fullPathOfResponse> ")


def main():
    if len(sys.argv) != 5:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    print(f'Program ID of Predictive Network training:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'Training log is in {trainLogFile}')
    print(f'.........')

    logging.info(f'Program ID of SkyWatcher Network training:{os.getpid()}\n')
    logging.info(f'Program command: \n {sys.argv}')
    logging.info(logNotes)

    curTime = datetime.datetime.now()
    logging.info(f'\nProgram starting Time: {str(curTime)}')

    netPath = sys.argv[1]
    trainingInputsPath = sys.argv[2]
    testInputsPath = sys.argv[3]
    responsePath = sys.argv[4]
    inputSuffix = "_roi.npy"

    Kr = 2  # treatment response 1 or 0
    Kup = 3  # segmentation classification number

    logging.info(f"Info: netPath = {netPath}\n")

    mergeTrainTestData = False

    trainDataMgr = Image3dResponseDataMgr(trainingInputsPath, responsePath, inputSuffix, logInfoFun=logging.info)

    if not mergeTrainTestData:
        testDataMgr = Image3dResponseDataMgr(testInputsPath, responsePath, inputSuffix, logInfoFun=logging.info)
    else:
        trainDataMgr.expandInputsDir(testInputsPath, suffix=inputSuffix)
        trainDataMgr.initializeInputsResponseList()

    # ===========debug==================
    trainDataMgr.setOneSampleTraining(True)  # for debug
    if not mergeTrainTestData:
        testDataMgr.setOneSampleTraining(True)  # for debug
    useDataParallel = True  # for debug
    # ===========debug==================


    batchSize = 4
    C = 24   # number of channels after the first input layer
    D = 147  # depth of input
    H = 281  # height of input
    W = 281  # width of input
    nDownSamples = 6


    trainDataMgr.setDataSize(batchSize, D, H, W, Kr, "TrainData")
    # batchSize, depth, height, width, k, # do not consider lymph node with label 3
    if not mergeTrainTestData:
        testDataMgr.setDataSize(batchSize, D, H, W, Kr, "TestData")  # batchSize, depth, height, width, k

    net = SkyWatcherModel(C, Kr, Kup, (D, H, W), nDownSamples)
    logging.info(f"Info: the size of bottle neck in the net = {C}* {net.m_bottleNeckSize}\n")

    trainDataMgr.setMixup(alpha=0.4, prob=0.5)  # set Mixup parameters

    optimizer = optim.Adam(net.parameters())
    net.setOptimizer(optimizer)

    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30, min_lr=1e-9)

    # Load network
    netMgr = NetMgr(net, netPath)
    bestTestPerf = 0
    if 2 == len(trainDataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")  # True for train
        logging.info(f'Program loads net from {netPath}.')
        bestTestPerf = netMgr.loadBestTestPerf()
        logging.info(f'Current best test dice: {bestTestPerf}')
    else:
        logging.info(f"Network trains from scratch.")

    logging.info(net.getParametersScale())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # lossFunc0 is for treatment response
    responseCEWeight = torch.FloatTensor(trainDataMgr.getResponseCEWeight()).to(device)
    responseFocalLoss = FocalCELoss(weight=responseCEWeight)
    net.appendLossFunc(responseFocalLoss, 1)

    # lossFunc1 and lossFunc2 are for segmentation.
    # After 100 epochs, we need to change foclas and segBoundaryLoss to 0.32: 0.68
    segCEWeight = torch.FloatTensor(trainDataMgr.getSegCEWeight()).to(device)
    segFocalLoss = FocalCELoss(weight=segCEWeight)
    segBoundaryLoss = BoundaryLoss(lambdaCoeff=0.001, k=Kup, weight=segCEWeight)
    net.appendLossFunc(segFocalLoss, 1)
    net.appendLossFunc(segBoundaryLoss, 0)
    # ========= end of loss function =================

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

    epochs = 1500000
    logging.info(f"Hints: Optimal_Result = Yes = 1,  Optimal_Result = No = 0 \n\n")

    logging.info(
        f"Epoch\t\tTrLoss\t" + "TrainAccuracy" + f"\t" + f"TsLoss\t" + f"TestAccuracy")  # logging.info output head

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

        if 105 == epoch:
            lossWeightList[0] = 1     # for response cross entropy
            lossWeightList[1] = 0.32  # for seg Cross Entropy
            lossWeightList[2] = 0.68  # for seg boundary loss
            logging.info(f"before just epoch {epoch}, fix loss weight as {lossWeightList}")
            if useDataParallel:
                net.module.updateLossWeightList(lossWeightList)
            else:
                net.updateLossWeightList(lossWeightList)

        for (inputs1, seg1Cpu, response1Cpu), (inputs2, seg2Cpu, response2Cpu) in zip(trainDataMgr.dataSegResponseGenerator(True),
                                                                trainDataMgr.dataSegResponseGenerator(True)):
            lambdaInBeta = trainDataMgr.getLambdaInBeta()
            inputs = inputs1 * lambdaInBeta + inputs2 * (1 - lambdaInBeta)
            inputs = torch.from_numpy(inputs).to(device, dtype=torch.float)
            seg1 = torch.from_numpy(seg1Cpu).to(device, dtype=torch.long)
            seg2 = torch.from_numpy(seg2Cpu).to(device, dtype=torch.long)

            if useDataParallel:
                optimizer.zero_grad()
                outputs = net.forward(inputs)
                loss = torch.tensor(0.0).cuda()
                for lossFunc, weight in zip(net.module.m_lossFuncList, lossWeightList):
                    if weight == 0:
                        continue
                    if lambdaInBeta != 0:
                        loss += lossFunc(outputs, seg1) * weight * lambdaInBeta
                    if 1 - lambdaInBeta != 0:
                        loss += lossFunc(outputs, seg2) * weight * (1 - lambdaInBeta)
                loss.backward()
                optimizer.step()
                batchLoss = loss.item()
            else:
                batchLoss = net.batchTrainMixup(inputs, seg1, seg2, lambdaInBeta)

            if lambdaInBeta == 1:
                nTrainCorrect += seg1.eq(torch.argmax(outputs, dim=1)).sum().item()
                nTrainTotal += seg1.shape[0]

            trainingLoss += batchLoss
            trainBatches += 1

        if 0 != trainBatches and 0 != nTrainTotal:
            trainingLoss /= trainBatches
            trainAccuracy = nTrainCorrect / nTrainTotal

        # ================Test===============

        testLoss = 0.0
        testBatches = 0
        nTestCorrect = 0
        nTestTotal = 0
        testAccuracy = 0
        if not mergeTrainTestData:
            net.eval()
            with torch.no_grad():
                for inputs, labelsCpu in testDataMgr.dataResponseGenerator(True):
                    inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labelsCpu)
                    inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device,
                                                                                     dtype=torch.long)  # return a copy
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

                    nTestCorrect += labels.eq(torch.argmax(outputs, dim=1)).sum().item()
                    nTestTotal += labels.shape[0]

                    testLoss += batchLoss
                    testBatches += 1

                # ===========print train and test progress===============
                if 0 != testBatches and 0 != nTestTotal:
                    testLoss /= testBatches
                    testAccuracy = nTestCorrect / nTestTotal
                    lrScheduler.step(testLoss)
        else:
            lrScheduler.step(trainingLoss)

        logging.info(
            f'{epoch}\t\t{trainingLoss:.4f}\t' + f'{trainAccuracy:.5f}' + f'\t' + f'\t{testLoss:.4f}\t' + f'{testAccuracy:.5f}')

        # =============save net parameters==============
        if trainingLoss != float('inf') and trainingLoss != float('nan'):
            if mergeTrainTestData:
                netMgr.saveNet()
                if trainAccuracy > bestTestPerf:
                    bestTestPerf = trainAccuracy
                    netMgr.saveBest(bestTestPerf)

            else:
                netMgr.save(testAccuracy)
                if testAccuracy > bestTestPerf:
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
