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

from DataMgr import DataMgr
from SegV3DModel import SegV3DModel
from SegV2DModel import SegV2DModel
from NetMgr  import NetMgr
from CustomizedLoss import FocalCELoss,BoundaryLoss

import numpy as np

# you may need to change the file name and log Notes below for every training.
trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/Skip2_20190506.txt'''
logNotes = r'''
Major program changes: ConvSeqential use BatchNorm-reLU-Conv structure, 
                       and each block has 5 layers, 
                       Residual connect to each Conv, and skip at least 2 layers.
                       use boundary loss with weight 0 at beginning, and pretrain CE loss. 
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

    print(f'Program ID {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'log is in {trainLogFile}')
    print(f'.........')

    logging.info(f'Program ID {os.getpid()}\n')
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

    useMixup = False
    alpha = 0.4  # for Beta distribution
    if useMixup:
        logging.info(f"Info: program uses mixeup with alpha={alpha}.")


    logging.info(f"Info: netPath = {netPath}\n")

    trainDataMgr = DataMgr(imagesPath, labelsPath, logInfoFun=logging.info)
    testDataMgr = DataMgr(*trainDataMgr.getTestDirs(), logInfoFun=logging.info)
    trainDataMgr.setRemainedLabel(3, labelTuple)
    testDataMgr.setRemainedLabel(3, labelTuple)

    # ===========debug==================
    trainDataMgr.setOneSampleTraining(False)  # for debug
    testDataMgr.setOneSampleTraining(False)  # for debug
    useDataParallel = True  # for debug
    # ===========debug==================

    trainDataMgr.buildSegSliceTupleList()
    testDataMgr.buildSegSliceTupleList()

    if is2DInput:
        logging.info(f"Info: program uses 2D input.")
        trainDataMgr.setDataSize(8, 1, 281, 281, K, "TrainData")  # batchSize, depth, height, width, k, # do not consider lymph node with label 3
        testDataMgr.setDataSize(8, 1, 281, 281, K, "TestData")  # batchSize, depth, height, width, k
        if 2 in trainDataMgr.m_remainedLabels:
            net = SegV2DModel(192, K)  # when increase the number of filter in first layer, you may consider to reduce batchSize because of GPU memory limits.
        else:
            net = SegV2DModel(128, K)

    else:
        logging.info(f"Info: program uses 3D input.")
        trainDataMgr.setDataSize(4, 21, 281, 281, K, "TrainData")  # batchSize, depth, height, width, k
        testDataMgr.setDataSize(4, 21, 281, 281, K, "TestData")  # batchSize, depth, height, width, k
        net = SegV3DModel(K)

    trainDataMgr.setMaxShift(25, 0.5)             #translation data augmentation and its probability
    trainDataMgr.setFlipProb(0.3)                 #flip data augmentation
    trainDataMgr.setRot90sProb(0.3)               #rotate along 90, 180, 270
    # trainDataMgr.setJitterNoise(0.3, 1)           #add Jitter noise
    trainDataMgr.setAddedNoise(0.3, 0.0,  0.1)     #add gaussian noise augmentation after data normalization of [0,1]

    optimizer = optim.Adam(net.parameters())
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30, min_lr=1e-8)

    # Load network
    netMgr = NetMgr(net, netPath)
    bestTestDiceList = [0] * K
    if 2 == len(trainDataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet(True)  # True for train
        logging.info(f'Program loads net from {netPath}.')
        bestTestDiceList = netMgr.loadBestTestDice(K)
        logging.info(f'Current best test dice: {bestTestDiceList}')
    else:
        logging.info(f"Network trains from scratch.")

    logging.info(net.getParametersScale())
    if 2 in trainDataMgr.m_remainedLabels:
        logging.info(net.setDropoutProb(0.2))           # metastases is hard to learn, so it need a smaller dropout rate.
    else:
        logging.info(net.setDropoutProb(0.3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ceWeight = torch.FloatTensor(trainDataMgr.getCEWeight()).to(device)
    focalLoss = FocalCELoss(weight=ceWeight)
    net.appendLossFunc(focalLoss, 1)
    boundaryLoss = BoundaryLoss(lambdaCoeff=0.001)
    net.appendLossFunc(boundaryLoss, 0)

    # logging.info model
    logging.info(f"\n====================Net Architecture===========================")
    stdoutBackup = sys.stdout
    with open(trainLogFile,'a+') as log:
        sys.stdout = log
        summary(net.cuda(), trainDataMgr.getInputSize())
    sys.stdout = stdoutBackup
    logging.info(f"===================End of Net Architecture =====================\n")

    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU >1:
            logging.info(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net)
    net.to(device)

    if useDataParallel:
        logging.info(net.module.lossFunctionsInfo())
    else:
        logging.info(net.lossFunctionsInfo())

    epochs = 15000
    logging.info(f"Hints: Test Dice_0 is the dice coeff for all non-zero labels")
    logging.info(f"Hints: Test Dice_1 is for primary cancer(green), test Dice_2 is for metastasis(yellow), and test Dice_3 is for invaded lymph node(brown).")
    logging.info(f"Hints: Test TPR_0 is the TPR for all non-zero labels")
    logging.info(f"Hints: Test TPR_1 is for primary cancer(green), TPR_2 is for metastasis(yellow), and TPR_3 is for invaded lymph node(brown).\n")
    diceHead = (f'Dice_{i}' for i in labelTuple)
    TPRHead = (f'TPR_{i}' for i in labelTuple)
    logging.info(f"Epoch \t TrainingLoss \t TestLoss \t"+f'\t'.join(diceHead) +f'\t' + f'\t'.join(TPRHead))   # logging.info output head

    for epoch in range(epochs):

        #================Update Loss weight==============
        lossWeightList = net.module.getLossWeightList() if useDataParallel else net.getLossWeightList()

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
        trainingLoss = 0.0
        batches = 0
        net.train()
        if useDataParallel:
            lossWeightList = torch.Tensor(net.module.m_lossWeightList).to(device)

        if useMixup: # use MIXUP.
            for (inputs1, labels1), (inputs2, labels2) in zip(trainDataMgr.dataLabelGenerator(True), trainDataMgr.dataLabelGenerator(True)):
                lambdaInBeta = np.random.beta(alpha, alpha)
                inputs = inputs1* lambdaInBeta + inputs2*(1-lambdaInBeta)
                inputs = torch.from_numpy(inputs).to(device, dtype=torch.float)
                labels1= torch.from_numpy(labels1).to(device, dtype=torch.long)
                labels2 = torch.from_numpy(labels2).to(device, dtype=torch.long)

                if useDataParallel:
                    optimizer.zero_grad()
                    outputs = net.forward(inputs)
                    loss = torch.tensor(0.0).cuda()
                    for lossFunc, weight in zip(net.module.m_lossFuncList, lossWeightList):
                        loss += lossFunc(outputs, labels1) * weight*lambdaInBeta
                        loss += lossFunc(outputs, labels2) * weight * (1-lambdaInBeta)
                    loss.backward()
                    optimizer.step()
                    batchLoss = loss.item()
                else:
                    batchLoss = net.batchTrainMixup(inputs, labels1, labels2, lambdaInBeta)


                trainingLoss += batchLoss
                batches += 1
        else:  # DO NOT USE MIXUP
            for inputs, labels in trainDataMgr.dataLabelGenerator(True):
                inputs, labels = torch.from_numpy(inputs).to(device, dtype=torch.float), torch.from_numpy(labels).to(device,dtype=torch.long)

                if useDataParallel:
                    optimizer.zero_grad()
                    outputs = net.forward(inputs)
                    loss = torch.tensor(0.0).cuda()
                    for lossFunc, weight in zip(net.module.m_lossFuncList, lossWeightList):
                        loss += lossFunc(outputs, labels) * weight
                    loss.backward()
                    optimizer.step()
                    batchLoss = loss.item()
                else:
                    batchLoss = net.batchTrain(inputs, labels)

                trainingLoss += batchLoss
                batches += 1

        if 0 != batches:
            trainingLoss /= batches

        # ================Test===============
        net.eval()
        with torch.no_grad():
            diceSumList = [0 for _ in range(K)]
            diceCountList = [0 for _ in range(K)]
            TPRSumList = [0 for _ in range(K)]
            TPRCountList = [0 for _ in range(K)]
            testLoss = 0.0
            batches = 0
            for inputs, labelsCpu in testDataMgr.dataLabelGenerator(False):
                inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labelsCpu)
                inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)  # return a copy

                if useDataParallel:
                    outputs = net.forward(inputs)
                    loss = torch.tensor(0.0).cuda()
                    for lossFunc, weight in zip(net.module.m_lossFuncList, lossWeightList):
                        loss += lossFunc(outputs, labels) * weight
                    batchLoss = loss.item()
                else:
                    batchLoss, outputs = net.batchTest(inputs, labels)

                outputs = outputs.cpu().numpy()
                segmentations = testDataMgr.oneHotArray2Segmentation(outputs)
                
                (diceSumBatch, diceCountBatch) = testDataMgr.getDiceSumList(segmentations, labelsCpu)
                (TPRSumBatch, TPRCountBatch) = testDataMgr.getTPRSumList(segmentations, labelsCpu)
                
                diceSumList = [x+y for x,y in zip(diceSumList, diceSumBatch)]
                diceCountList = [x+y for x,y in zip(diceCountList, diceCountBatch)]
                TPRSumList = [x + y for x, y in zip(TPRSumList, TPRSumBatch)]
                TPRCountList = [x + y for x, y in zip(TPRCountList, TPRCountBatch)]
                
                testLoss += batchLoss
                batches += 1
                #logging.info(f'batch={batches}: batchLoss = {batchLoss}')

        #===========print train and test progress===============
        if 0 != batches:
            testLoss /= batches
            lrScheduler.step(testLoss)
        diceAvgList = [x/(y+1e-8) for x,y in zip(diceSumList, diceCountList)]
        TPRAvgList = [x / (y + 1e-8) for x, y in zip(TPRSumList, TPRCountList)]
        logging.info(f'{epoch} \t {trainingLoss:.4f} \t {testLoss:.4f} \t'+f'\t'.join( (f'{x:.3f}' for x in diceAvgList))+f'\t'+f'\t'.join( (f'{x:.3f}' for x in TPRAvgList)))

        # =============save net parameters==============
        if trainingLoss != float('inf') and trainingLoss != float('nan'):
            netMgr.save(diceAvgList)
            if diceAvgList[1] > 0.20  and diceAvgList[1] > bestTestDiceList[1]:  # compare the primary dice.
                bestTestDiceList = diceAvgList
                netMgr.saveBest(bestTestDiceList)
        else:
            logging.info(f"Error: training loss is infinity. Program exit.")
            sys.exit()

    torch.cuda.empty_cache()
    logging.info(f"=============END of Training of Ovarian Cancer Segmentation V Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')

if __name__ == "__main__":
    main()
