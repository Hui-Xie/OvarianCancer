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
trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/log_SkyWatcher_20190606.txt'''
logNotes = r'''
Major program changes: 
                     delete the m_k in the DataMgr class.
                      

Experiment setting for Image3d ROI to response:
Input CT data: 147*281*281  3D CT raw image ROI
segmentation label: 127*255*255 segmentation label with value (0,1,2) which erases lymph node label

This is a multi-task learning. 

Predictive Model: 1,  first 3-layer dense conv block with channel size 6.
                  2,  and 6 dense conv DownBB blocks,  each of which includes a stride 2 conv and 3-layers dense conv block; 
                  3,  and 3 fully connected layers  changes the tensor into size 2*1;
                  4,  final a softmax for binary classification;
                  Total network learning parameters are 25K.
                  Network architecture is referred at https://github.com/Hui-Xie/OvarianCancer/blob/master/Image3dPredictModel.py

response Loss Function:   focus loss  with weight [3.3, 1.4] for [0,1] class separately, as [0,1] uneven distribution.
segmentation loss function: focus loss  with weight [1.0416883685076772, 39.37007874015748, 68.39945280437757] for label (0, 1, 2)

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
    trainDataMgr.setOneSampleTraining(False)  # for debug
    if not mergeTrainTestData:
        testDataMgr.setOneSampleTraining(False)  # for debug
    useDataParallel = True  # for debug
    outputTrainDice = True
    # ===========debug==================


    batchSize = 4
    C = 6   # number of channels after the first input layer
    D = 147  # depth of input
    H = 281  # height of input
    W = 281  # width of input
    nDownSamples = 6


    trainDataMgr.setDataSize(batchSize, D, H, W, "TrainData")
    # batchSize, depth, height, width, and do not consider lymph node with label 3
    if not mergeTrainTestData:
        testDataMgr.setDataSize(batchSize, D, H, W, "TestData")  # batchSize, depth, height, width

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
    segFocalLoss = FocalCELoss(weight=segCEWeight, ignore_index=-100) # ignore all zero slices
    net.appendLossFunc(segFocalLoss, 1)

    # boundaryLoss does not support 3D input.
    # segBoundaryLoss = BoundaryLoss(lambdaCoeff=0.001, k=Kup, weight=segCEWeight)
    # net.appendLossFunc(segBoundaryLoss, 0)
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
    logging.info(f"Hints: Test Dice_0 is the dice coeff for all non-zero labels")
    logging.info(
        f"Hints: Test Dice_1 is for primary cancer(green), \t\n test Dice_2 is for metastasis(yellow), \t\n and test Dice_3 is for invaded lymph node(brown).")
    logging.info(f"Hints: Test TPR_0 is the TPR for all non-zero labels")
    logging.info(
        f"Hints: Test TPR_1 is for primary cancer(green), \t\n TPR_2 is for metastasis(yellow), \t\n and TPR_3 is for invaded lymph node(brown).\n")
    logging.info(f"Dice is based on all 2D segmented slices in the volume from weak annotation, not real 3D dice.")
    diceHead1 = (f'Dice{i}' for i in range(Kup))  # generator object can be use only once.
    TPRHead1 = (f'TPR_{i}' for i in range(Kup))
    diceHead2 = (f'Dice{i}' for i in range(Kup))
    TPRHead2 = (f'TPR_{i}' for i in range(Kup))

    logging.info(f"\nHints: Optimal_Result = Yes = 1,  Optimal_Result = No = 0 \n")


    logging.info(f"Epoch\tTrLoss\t" + f"\t".join(diceHead1) + f"\t" + f"\t".join(TPRHead1) + f"\tAccura"\
                 + f"\tTsLoss\t" + f"\t".join(diceHead2) + f"\t" + f"\t".join(TPRHead2) + f"\tAccura")  # logging.info output head



    for epoch in range(epochs):
        # ================Training===============
        net.train()
        random.seed()

        trainingLoss = 0.0
        trainBatches = 0

        nTrainCorrect = 0
        nTrainTotal = 0
        trainAccuracy = 0

        trainDiceSumList = [0 for _ in range(Kup)]
        trainDiceCountList = [0 for _ in range(Kup)]
        trainTPRSumList = [0 for _ in range(Kup)]
        trainTPRCountList = [0 for _ in range(Kup)]


        if useDataParallel:
            lossWeightList = torch.Tensor(net.module.m_lossWeightList).to(device)
        else:
            lossWeightList = torch.Tensor(net.m_lossWeightList).to(device)

        for (inputs1, seg1Cpu, response1Cpu), (inputs2, seg2Cpu, response2Cpu) in zip(trainDataMgr.dataSegResponseGenerator(True),
                                                                                      trainDataMgr.dataSegResponseGenerator(True)):
            lambdaInBeta = trainDataMgr.getLambdaInBeta()
            inputs = inputs1 * lambdaInBeta + inputs2 * (1 - lambdaInBeta)
            inputs = torch.from_numpy(inputs).to(device, dtype=torch.float)
            seg1 = torch.from_numpy(seg1Cpu).to(device, dtype=torch.long)
            seg2 = torch.from_numpy(seg2Cpu).to(device, dtype=torch.long)
            response1 = torch.from_numpy(response1Cpu).to(device, dtype=torch.long)
            response2 = torch.from_numpy(response2Cpu).to(device, dtype=torch.long)

            if useDataParallel:
                optimizer.zero_grad()
                xr, xup = net.forward(inputs)
                loss = torch.tensor(0.0).cuda()

                for i, (lossFunc, weight) in enumerate(zip(net.module.m_lossFuncList, lossWeightList)):
                    if i ==0:
                        outputs = xr
                        gt1, gt2 = (response1, response2)
                    else:
                        outputs = xup
                        gt1, gt2 = (seg1, seg2)

                    if weight == 0:
                        continue
                    if lambdaInBeta != 0:
                        loss += lossFunc(outputs, gt1) * weight * lambdaInBeta
                    if 1 - lambdaInBeta != 0:
                        loss += lossFunc(outputs, gt2) * weight * (1 - lambdaInBeta)
                loss.backward()
                optimizer.step()
                batchLoss = loss.item()
            else:
                logging.info(f"SkyWatcher Training must use Dataparallel. Program exit.")
                sys.exit(-5)

            # compute response accuracy
            if lambdaInBeta == 1:
                nTrainCorrect += response1.eq(torch.argmax(xr, dim=1)).sum().item()
                nTrainTotal += response1.shape[0]
            if lambdaInBeta == 0:
                nTrainCorrect += response2.eq(torch.argmax(xr, dim=1)).sum().item()
                nTrainTotal += response2.shape[0]

            # compute segmentation dice and TPR
            if lambdaInBeta == 1 and outputTrainDice and epoch % 5 == 0:
                trainDiceSumList, trainDiceCountList, trainTPRSumList, trainTPRCountList \
                    = trainDataMgr.updateDiceTPRSumList(xup, seg1Cpu, Kup, trainDiceSumList, trainDiceCountList, trainTPRSumList, trainTPRCountList)
            if lambdaInBeta == 0 and outputTrainDice and epoch % 5 == 0:
                trainDiceSumList, trainDiceCountList, trainTPRSumList, trainTPRCountList \
                    = trainDataMgr.updateDiceTPRSumList(xup, seg2Cpu, Kup, trainDiceSumList, trainDiceCountList, trainTPRSumList, trainTPRCountList)

            trainingLoss += batchLoss
            trainBatches += 1

        if 0 != trainBatches and 0 != nTrainTotal:
            trainingLoss /= trainBatches
            trainAccuracy = nTrainCorrect / nTrainTotal

        trainDiceAvgList = [x / (y + 1e-8) for x, y in zip(trainDiceSumList, trainDiceCountList)]
        trainTPRAvgList = [x / (y + 1e-8) for x, y in zip(trainTPRSumList, trainTPRCountList)]

        # ================Test===============
        net.eval()

        testLoss = 0.0
        testBatches = 0

        nTestCorrect = 0
        nTestTotal = 0
        testAccuracy = 0

        testDiceSumList = [0 for _ in range(Kup)]
        testDiceCountList = [0 for _ in range(Kup)]
        testTPRSumList = [0 for _ in range(Kup)]
        testTPRCountList = [0 for _ in range(Kup)]

        if not mergeTrainTestData:

            with torch.no_grad():
                for inputs, segCpu, responseCpu in testDataMgr.dataSegResponseGenerator(True):
                    inputs, seg, response = torch.from_numpy(inputs), torch.from_numpy(segCpu), torch.from_numpy(responseCpu)
                    inputs, seg, response = inputs.to(device, dtype=torch.float), seg.to(device, dtype=torch.long), response.to(device, dtype=torch.long)  # return a copy
                    if useDataParallel:
                        xr, xup = net.forward(inputs)
                        loss = torch.tensor(0.0).cuda()

                        for i, (lossFunc, weight) in enumerate(zip(net.module.m_lossFuncList, lossWeightList)):
                            if i == 0:
                                outputs = xr
                                gt = response
                            else:
                                outputs = xup
                                gt = seg

                            if weight != 0:
                               loss += lossFunc(outputs, gt) * weight

                        batchLoss = loss.item()
                    else:
                        logging.info(f"SkyWatcher Test must use Dataparallel. Program exit.")
                        sys.exit(-5)

                    nTestCorrect += response.eq(torch.argmax(xr, dim=1)).sum().item()
                    nTestTotal += response.shape[0]

                    testDiceSumList, testDiceCountList, testTPRSumList, testTPRCountList \
                        = testDataMgr.updateDiceTPRSumList(xup, segCpu, Kup, testDiceSumList, testDiceCountList, testTPRSumList, testTPRCountList)

                    testLoss += batchLoss
                    testBatches += 1

                # ===========print train and test progress===============
                if 0 != testBatches and 0 != nTestTotal:
                    testLoss /= testBatches
                    testAccuracy = nTestCorrect / nTestTotal
                    lrScheduler.step(testLoss)
        else:
            lrScheduler.step(trainingLoss)

        testDiceAvgList = [x / (y + 1e-8) for x, y in zip(testDiceSumList, testDiceCountList)]
        testTPRAvgList = [x / (y + 1e-8) for x, y in zip(testTPRSumList, testTPRCountList)]

        logging.info(
            f'{epoch}\t{trainingLoss:.4f}\t' + f'\t'.join((f'{x:.3f}' for x in trainDiceAvgList)) + f'\t' + f'\t'.join((f'{x:.3f}' for x in trainTPRAvgList)) +  f'\t{trainAccuracy:.4f}'\
            + f'\t{testLoss:.4f}\t' + f'\t'.join((f'{x:.3f}' for x in testDiceAvgList)) + f'\t' + f'\t'.join((f'{x:.3f}' for x in testTPRAvgList)) + f'\t{testAccuracy:.4f}')

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
    logging.info(f"=============END of Training of SkyWatcher Predict Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')


if __name__ == "__main__":
    main()
