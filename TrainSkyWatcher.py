#  train Skywatcher Model for segmentation and treatment response together

import sys
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import numpy as np

from Image3dResponseDataMgr import Image3dResponseDataMgr
from SkyWatcherModel2 import SkyWatcherModel2
from NetMgr import NetMgr
from CustomizedLoss import FocalCELoss, BoundaryLoss

# you may need to change the file name and log Notes below for every training.
trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/log_SkyWatcher_CV01_20190626.txt'''
#trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/log_temp_20190624.txt'''
logNotes = r'''
Major program changes: 
                      merge train and test imageDataMgr into one.
                      when epoch %5 ==0, do not use mixup.
                      And Only when epoch %5 ==0, print log and save.
                      use batchSize = 9, and 3GPU  training.
                      along deeper layer, increase filter number.
                      put all data into local GPU sever.
                      10 fold cross validation, 0 fold for test.
                      data partition with patient ID, instead of VOI.
                      in image3dResponseDataMgr, random Crop ROI in the fly.
                      erase normalization layers  in the fully connected layers.
                      

Experiment setting for Image3d ROI to response:
Input CT data: 29*140*140  3D CT raw image ROI with spacing size(5*2*2)
segmentation label: 23*127*127 with spacing size(5*2*2) segmentation label with value (0,1,2) which erases lymph node label

This is a multi-task learning. 

Predictive Model: 1,  first 3-layer dense conv block with channel size 128.
                  2,  and 3 dense conv DownBB blocks,  each of which includes a stride 2 conv and 3-layers dense conv block; 
                  3,  and 3 fully connected layers  changes the tensor into size 2*1;
                  4,  final a softmax for binary classification;
                  Total network learning parameters are 8 million.
                  Network architecture is referred at https://github.com/Hui-Xie/OvarianCancer/blob/master/SkyWatcherModel.py

response Loss Function:   focus loss  with weight [3.3, 1.4] for [0,1] class separately, as [0,1] uneven distribution.
segmentation loss function: focus loss  with weight [1.0416883685076772, 39.37007874015748, 68.39945280437757] for label (0, 1, 2)

Data:   training data has 113 patients, and valdiation data has 27 patients with training/test rate 80/20.
        We randomize all data, and then assign same distrubtion of treat reponse 0,1 into to training and test data set.
        

Training strategy:  50% probability of data are mixed up with beta distribution with alpha =0.4, to feed into network for training. 
                    No other data augmentation, and no dropout.  

                    Learning Scheduler:  Reduce learning rate on  plateau, and learning rate patience is 30 epochs.                                

            '''

logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')


def printUsage(argv):
    print("============Train SkyWatcher Model for Ovarian Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <fullPathOfData>  <fullPathOfLabels> <fullPathOfResponseFile> ")


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
    dataInputsPath = sys.argv[2]
    labelInputsPath = sys.argv[3]  # may not use.
    responsePath = sys.argv[4]
    inputSuffix = ".npy"

    Kr = 2  # treatment response 1 or 0
    Kup = 3  # segmentation classification number
    K_fold = 10
    k = 0
    logging.info(f"Info: this is the {k}th fold leave for test in the {K_fold}-fold cross-validation.\n")

    logging.info(f"Info: netPath = {netPath}\n")

    mergeTrainTestData = False

    dataMgr = Image3dResponseDataMgr(dataInputsPath, responsePath, inputSuffix, K_fold, k, logInfoFun=logging.info)

    # ===========debug==================
    dataMgr.setOneSampleTraining(True)  # for debug
    useDataParallel = True  # for debug
    GPU_ID = 0  # choices: 0,1,2,3 for lab server.
    # ===========debug==================


    batchSize = 6
    C = 32   # number of channels after the first input layer
    D = 29  # depth of input
    H = 140  # height of input
    W = 140  # width of input

    dataMgr.setDataSize(batchSize, D, H, W, "TrainTestData")
    # batchSize, depth, height, width, and do not consider lymph node with label 3

    net = SkyWatcherModel2(C, Kr, Kup, (D, H, W))
    net.apply(net.initializeWeights)
    logging.info(f"Info: the size of bottle neck in the net = {net.m_bottleNeckSize}\n")

    dataMgr.setMixup(alpha=0.4, prob=0.5)  # set Mixup parameters

    optimizer = optim.Adam(net.parameters())
    net.setOptimizer(optimizer)

    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=30, min_lr=1e-9)

    # Load network
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    net.to(device)
    netMgr = NetMgr(net, netPath, device)


    bestTestPerf = 0
    if 2 == len(dataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")  # True for train
        logging.info(f'Program loads net from {netPath}.')
        bestTestPerf = netMgr.loadBestTestPerf()
        logging.info(f'Current best test dice: {bestTestPerf}')
    else:
        logging.info(f"Network trains from scratch.")

    logging.info(net.getParametersScale())




    # lossFunc0 is for treatment response
    responseCEWeight = torch.FloatTensor(dataMgr.getResponseCEWeight()).to(device)
    responseFocalLoss = FocalCELoss(weight=responseCEWeight)
    net.appendLossFunc(responseFocalLoss, 1)

    # lossFunc1 and lossFunc2 are for segmentation.
    # After 100 epochs, we need to change foclas and segBoundaryLoss to 0.32: 0.68
    segCEWeight = torch.FloatTensor(dataMgr.getSegCEWeight()).to(device)
    segFocalLoss = FocalCELoss(weight=segCEWeight, ignore_index=-100) # ignore all zero slices
    net.appendLossFunc(segFocalLoss, 1)

    # boundaryLoss does not support 3D input.
    # segBoundaryLoss = BoundaryLoss(lambdaCoeff=0.001, k=Kup, weight=segCEWeight)
    # net.appendLossFunc(segBoundaryLoss, 0)
    # ========= end of loss function =================


    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU > 1:
            device_ids = [1, 2, 3]
            logging.info(f'Info: program will use {len(device_ids)} GPUs.')
            net = nn.DataParallel(net, device_ids=[0,1], output_device=device)

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


    logging.info(f"Epoch\tTrLoss" + f"\t" + f"\t".join(diceHead1) + f"\t" + f"\t".join(TPRHead1) + f"\tAccura" + f"\tTPR_r" +  f"\tTNR_r"\
                 + f"\t\tTsLoss"  + f"\t" + f"\t".join(diceHead2) + f"\t" + f"\t".join(TPRHead2) + f"\tAccura" + f"\tTPR_r" +  f"\tTNR_r")  # logging.info output head

    oldTestLoss = 1000

    for epoch in range(epochs):
        # ================Training===============
        net.train()
        random.seed()

        trainingLoss = 0.0
        trainBatches = 0

        epochPredict = None
        epochResponse = None
        responseTrainAccuracy = 0.0
        responseTrainTPR = 0.0
        responseTrainTNR = 0.0

        trainDiceSumList = [0 for _ in range(Kup)]
        trainDiceCountList = [0 for _ in range(Kup)]
        trainTPRSumList = [0 for _ in range(Kup)]
        trainTPRCountList = [0 for _ in range(Kup)]


        if useDataParallel:
            lossWeightList = torch.Tensor(net.module.m_lossWeightList).to(device)
        else:
            lossWeightList = torch.Tensor(net.m_lossWeightList).to(device)

        for (inputs1, seg1Cpu, response1Cpu), (inputs2, seg2Cpu, response2Cpu) in zip(dataMgr.dataSegResponseGenerator(dataMgr.m_trainingSetIndices, shuffle=True),
                                                                                      dataMgr.dataSegResponseGenerator(dataMgr.m_trainingSetIndices, shuffle=True)):
            if epoch % 5 == 0:
                lambdaInBeta = 1                          # this will make the comparison in the segmention per 5 epochs meaningful.
            else:
                lambdaInBeta = dataMgr.getLambdaInBeta()

            inputs = inputs1 * lambdaInBeta + inputs2 * (1 - lambdaInBeta)
            inputs = torch.from_numpy(inputs).to(device, dtype=torch.float)
            seg1 = torch.from_numpy(seg1Cpu).to(device, dtype=torch.long)
            seg2 = torch.from_numpy(seg2Cpu).to(device, dtype=torch.long)
            response1 = torch.from_numpy(response1Cpu).to(device, dtype=torch.long)
            response2 = torch.from_numpy(response2Cpu).to(device, dtype=torch.long)


            optimizer.zero_grad()
            xr, xup = net.forward(inputs)
            loss = torch.tensor(0.0).to(device)

            for i, (lossFunc, weight) in enumerate(zip(net.module.m_lossFuncList if useDataParallel else net.m_lossFuncList,
                                                       lossWeightList)):
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


            # accumulate response and predict value
            if epoch % 5 == 0:
                batchPredict = torch.argmax(xr, dim=1).cpu().detach().numpy().flatten()
                epochPredict = np.concatenate((epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                epochResponse = np.concatenate((epochResponse, response1Cpu)) if epochResponse is not None else response1Cpu
                trainDiceSumList, trainDiceCountList, trainTPRSumList, trainTPRCountList \
                                         = dataMgr.updateDiceTPRSumList(xup, seg1Cpu, Kup, trainDiceSumList, trainDiceCountList,
                                                                                           trainTPRSumList, trainTPRCountList)

            trainingLoss += batchLoss
            trainBatches += 1

        if 0 != trainBatches:
            trainingLoss /= trainBatches

        if epoch % 5 == 0:
            responseTrainAccuracy = dataMgr.getAccuracy(epochPredict, epochResponse)
            responseTrainTPR = dataMgr.getTPR(epochPredict, epochResponse)[0]
            responseTrainTNR = dataMgr.getTNR(epochPredict, epochResponse)[0]
            trainDiceAvgList = [x / (y + 1e-8) for x, y in zip(trainDiceSumList, trainDiceCountList)]
            trainTPRAvgList = [x / (y + 1e-8) for x, y in zip(trainTPRSumList, trainTPRCountList)]
        else:
            continue  # only epoch %5 ==0, run validation set.

        # ================Test===============
        net.eval()

        testLoss = 0.0
        testBatches = 0

        epochPredict = None
        epochResponse = None
        responseTestAccuracy = 0.0
        responseTestTPR = 0.0
        responseTestTNR = 0.0

        testDiceSumList = [0 for _ in range(Kup)]
        testDiceCountList = [0 for _ in range(Kup)]
        testTPRSumList = [0 for _ in range(Kup)]
        testTPRCountList = [0 for _ in range(Kup)]

        if not mergeTrainTestData:

            with torch.no_grad():
                for inputs, segCpu, responseCpu in dataMgr.dataSegResponseGenerator(dataMgr.m_validationSetIndices, shuffle=True, randomROI=False):
                    inputs, seg, response = torch.from_numpy(inputs), torch.from_numpy(segCpu), torch.from_numpy(responseCpu)
                    inputs, seg, response = inputs.to(device, dtype=torch.float), seg.to(device, dtype=torch.long), response.to(device, dtype=torch.long)  # return a copy

                    xr, xup = net.forward(inputs)
                    loss = torch.tensor(0.0).to(device)

                    for i, (lossFunc, weight) in enumerate(zip(net.module.m_lossFuncList if useDataParallel else net.m_lossFuncList,
                                                               lossWeightList)):
                        if i == 0:
                            outputs = xr
                            gt = response
                        else:
                            outputs = xup
                            gt = seg

                        if weight != 0:
                           loss += lossFunc(outputs, gt) * weight

                    batchLoss = loss.item()


                    # accumulate response and predict value

                    batchPredict = torch.argmax(xr, dim=1).cpu().detach().numpy().flatten()
                    epochPredict = np.concatenate((epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                    epochResponse = np.concatenate((epochResponse, responseCpu)) if epochResponse is not None else responseCpu

                    testDiceSumList, testDiceCountList, testTPRSumList, testTPRCountList \
                        = dataMgr.updateDiceTPRSumList(xup, segCpu, Kup, testDiceSumList, testDiceCountList, testTPRSumList, testTPRCountList)

                    testLoss += batchLoss
                    testBatches += 1

                # ===========print train and test progress===============
                if 0 != testBatches:
                    testLoss /= testBatches
                    lrScheduler.step(testLoss)

                if epoch % 5 == 0:
                    responseTestAccuracy = dataMgr.getAccuracy(epochPredict, epochResponse)
                    responseTestTPR = dataMgr.getTPR(epochPredict, epochResponse)[0]
                    responseTestTNR = dataMgr.getTNR(epochPredict, epochResponse)[0]

        else:
            lrScheduler.step(trainingLoss)


        testDiceAvgList = [x / (y + 1e-8) for x, y in zip(testDiceSumList, testDiceCountList)]
        testTPRAvgList  = [x / (y + 1e-8) for x, y in zip(testTPRSumList, testTPRCountList)]

        outputString =  f'{epoch}\t{trainingLoss:.4f}\t' + f'\t'.join((f'{x:.3f}' for x in trainDiceAvgList)) + f'\t' + f'\t'.join((f'{x:.3f}' for x in trainTPRAvgList)) + f'\t{responseTrainAccuracy:.4f}' + f'\t{responseTrainTPR:.4f}' + f'\t{responseTrainTNR:.4f}'
        outputString += f'\t\t{testLoss:.4f}\t' + f'\t'.join((f'{x:.3f}' for x in testDiceAvgList)) + f'\t' + f'\t'.join((f'{x:.3f}' for x in testTPRAvgList)) + f'\t{responseTestAccuracy:.4f}'+f'\t{responseTestTPR:.4f}'+ f'\t{responseTestTNR:.4f}'
        logging.info(outputString)

        # =============save net parameters==============
        if trainingLoss != float('inf') and trainingLoss != float('nan'):
            if mergeTrainTestData:
                netMgr.saveNet()
                if responseTrainAccuracy >= bestTestPerf and trainingLoss < oldTrainingLoss:
                    oldTrainingLoss = trainingLoss
                    bestTestPerf = responseTrainAccuracy
                    netMgr.saveBest(bestTestPerf)

            else:
                netMgr.save(responseTestAccuracy)
                if responseTestAccuracy >= bestTestPerf and testLoss < oldTestLoss:
                    oldTestLoss = testLoss
                    bestTestPerf = responseTestAccuracy
                    netMgr.saveBest(bestTestPerf)
        else:
            logging.info(f"Error: training loss is infinity. Program exit.")
            sys.exit()

    torch.cuda.empty_cache()
    logging.info(f"=============END of Training of SkyWatcher Predict Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')


if __name__ == "__main__":
    main()
