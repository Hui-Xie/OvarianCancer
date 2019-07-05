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
import math

from Image3dResponseDataMgr import Image3dResponseDataMgr
from SkyWatcherModel2 import SkyWatcherModel2
from NetMgr import NetMgr
from CustomizedLoss import FocalCELoss, BoundaryLoss

logNotes = r'''
Major program changes: 
                      along deeper layer, increase filter number.
                      10 fold cross validation, 0 fold for test.
                      data partition with patient ID, instead of VOI.
                      in image3dResponseDataMgr, random Crop ROI in the fly.
                      erase all normalization layers in the fully connected layers.
                      Crop ROI around the mass center in each labeled slice. 
                      use reSampleForSameDistribution in training set, but keep original ditribution in the test set
                      First implement 1000 epochs in the segmentation path, and then freeze the encoder and decoder, only train the ResponseBranch.  
                      epoch < 1000, the loss is pure segmentation loss;
                      add FC layer width = 256*49 at first FC layer, and halves along deeper FC layer.
                      add  dropout of 0.5 in FC layers.
                      add data window level adjust, slice Normalization, gausssian noise, random flip, 90/180/270 rotation. 
                      reset learning rate patience after 1000 epochs.
                      disable data augmentation in the validation data;
                      in response prediction path, learning rate decacy patience set as 200 instead of 30.
                      when disable data augmentation, choose the fixed center labeled slice from a patient.
                      epoch >= 1000,  training Encoder and FC branch, and freeze decode. this training is base on  log_SkyWatcher_CV0_20190704_2129.txt
                                                    
 
Discarded changes:                      
                      training response branch per 5 epoch after epoch 100, while continuing train the segmenation branch.
                      which means that before epoch 100, the accuray data is a mess.
                      Add L2 norm regularization in Adam optimizer with weight 5e-4. 
                      Add dropout at Fully connected layer with dropout rate of 50%.  
                      focus loss  with weight [3.3, 1.4] for [0,1] class separately, as [0,1] uneven distribution.   
                       the loss is pure response loss with reinitialized learning rate 1e-3.
                      

Experiment setting for Image3d ROI to response:
Input CT data: 29*140*140  3D CT raw image ROI with spacing size(5*2*2)
segmentation label: 23*127*127 with spacing size(5*2*2) segmentation label with value (0,1,2) which erases lymph node label


Predictive Model: 1,  first 3-layer dense conv block with channel size 128.
                  2,  and 3 dense conv DownBB blocks,  each of which includes a stride 2 conv and 3-layers dense conv block; 
                  3,  and 3 fully connected layers  changes the tensor into size 2*1;
                  4,  final a softmax for binary classification;
                  Total network learning parameters are 119 million.
                  Network architecture is referred at https://github.com/Hui-Xie/OvarianCancer/blob/master/SkyWatcherModel2.py

response Loss Function:  focus loss with weight 1:1 as training data use response balance distribution with resample with replacement. 
segmentation loss function: focus loss  with weight [1.0416883685076772, 39.37007874015748, 68.39945280437757] for label (0, 1, 2)

Data:   training data has 153 patients, and valdiation data has 16 patients, under 10 fold partition.
        We randomize all data, and then assign same distrubtion of treat reponse 0,1 into to training set;
        Validation data set keep original distribution
        

Training strategy:  50% probability of data are mixed up with beta distribution with alpha =0.4, to feed into network for training. 
                    No other data augmentation, and no dropout.  

                    Learning Scheduler for segmentation:  Reduce learning rate on  plateau, and learning rate patience is 300 epochs.
                                                    
                    Learning Scheduler for response:      Reduce learning rate on  plateau, and learning rate patience is 30 epochs.
            '''

def printUsage(argv):
    print("============Train SkyWatcher Model for Ovarian Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <fullPathOfData>  <fullPathOfLabels> <fullPathOfResponseFile> k ")
    print("where: k=0-9, the k-th fold in the 10-fold cross validation.")

def main():
    if len(sys.argv) != 6:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    dataInputsPath = sys.argv[2]
    labelInputsPath = sys.argv[3]  # may not use.
    responsePath = sys.argv[4]
    k = int(sys.argv[5])
    inputSuffix = ".npy"

    curTime = datetime.datetime.now()
    trainLogFile = f'/home/hxie1/Projects/OvarianCancer/trainLog/log_SkyWatcher_CV{k:d}_{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}.txt'
    logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')

    print(f'Program ID of Predictive Network training:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'Training log is in {trainLogFile}')
    print(f'.........')

    logging.info(f'Program ID of SkyWatcher Network training:{os.getpid()}\n')
    logging.info(f'Program command: \n {sys.argv}')
    logging.info(logNotes)


    logging.info(f'\nProgram starting Time: {str(curTime)}')

    Kr = 2  # treatment response 1 or 0
    Kup = 3  # segmentation classification number
    K_fold = 10
    logging.info(f"Info: this is the {k}th fold leave for test in the {K_fold}-fold cross-validation.\n")

    logging.info(f"Info: netPath = {netPath}\n")

    dataMgr = Image3dResponseDataMgr(dataInputsPath, responsePath, inputSuffix, K_fold, k, logInfoFun=logging.info)
    dataMgr.setFlipProb(0.3)
    dataMgr.setRot90sProb(0.3)  # rotate along 90, 180, 270
    dataMgr.setAddedNoise(0.3, 0.0, 0.1)  # add gaussian noise augmentation after data normalization of [0,1]
    dataMgr.setMixup(alpha=0.4, prob=0.5)  # set Mixup parameters

    # ===========debug==================
    dataMgr.setOneSampleTraining(False)  # for debug
    useDataParallel = True  # for debug
    GPU_ID = 1  # choices: 0,1,2,3 for lab server.
    # ===========debug==================


    batchSize = 9  # 12 for use 4 GPUs
    C = 32   # number of channels after the first input layer
    D = 29  # depth of input
    H = 140  # height of input
    W = 140  # width of input

    dataMgr.setDataSize(batchSize, D, H, W, "TrainTestData")
    # batchSize, depth, height, width, and do not consider lymph node with label 3

    net = SkyWatcherModel2(C, Kr, Kup, (D, H, W))
    net.apply(net.initializeWeights)
    logging.info(f"Info: the size of bottle neck in the net = {net.m_bottleNeckSize}\n")



    optimizer = optim.Adam(net.parameters(), weight_decay=0)
    net.setOptimizer(optimizer)

    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=300, min_lr=1e-9)

    # Load network
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    net.to(device)
    netMgr = NetMgr(net, netPath, device)


    bestTestPerf = 0
    if 2 == len(dataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")  # True for train
        logging.info(f'Program loads net from {netPath}.')
        bestTestPerf = netMgr.loadBestTestPerf()
        logging.info(f'Current best test performance: {bestTestPerf}')
    else:
        logging.info(f"Network trains from scratch.")

    logging.info(net.getParametersScale())

    # lossFunc0 is for treatment response
    # responseCEWeight = torch.FloatTensor(dataMgr.getResponseCEWeight()).to(device)
    # responseFocalLoss = FocalCELoss(weight=responseCEWeight)
    responseFocalLoss = FocalCELoss() # use reSampleForSameDistribution in the response data, so we do not need weight.
    net.appendLossFunc(responseFocalLoss, 1)

    # lossFunc1 and lossFunc2 are for segmentation.
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
            net = nn.DataParallel(net, device_ids=device_ids, output_device=device)

    if useDataParallel:
        logging.info(net.module.lossFunctionsInfo())
    else:
        logging.info(net.lossFunctionsInfo())


    if useDataParallel:
        net.module.freezeResponseBranch(requires_grad=False)
        net.module.freezeSegmentationBranch(requires_grad=True)
    else:
        net.freezeResponseBranch(requires_grad=False)
        net.freezeSegmentationBranch(requires_grad=True)
    lrScheduler.patience = 300  # change learning patience

    pivotEpoch = 1000
    logging.info(f"when epoch < {pivotEpoch}, only train segmentation, which means response accuracy are meaningless at these epoch.")
    logging.info(f"when epoch >= {pivotEpoch}, only training response branch, which means segmentation accuracy should keep unchange.")


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

    oldTrainingLoss = 1000
    oldTestLoss = 1000

    for epoch in range(pivotEpoch, epochs):

        if epoch == pivotEpoch:
            if useDataParallel:
                net.module.freezeResponseBranch(requires_grad=True)
                net.module.freezeEncoder(requires_grad=True)
                net.module.freezeDecoder(requires_grad=False)

            else:
                net.freezeResponseBranch(requires_grad=True)
                net.freezeEncoder(requires_grad=True)
                net.freezeDecoder(requires_grad=False)

            # restore learning rate to initial value
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3
            lrScheduler.patience = 200  # change learning patience

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

        for (inputs1, seg1Cpu, response1Cpu), (inputs2, seg2Cpu, response2Cpu) in zip(dataMgr.dataSegResponseGenerator(dataMgr.m_trainingSetIndices, shuffle=True, dataAugment=True, reSample=True),
                                                                                      dataMgr.dataSegResponseGenerator(dataMgr.m_trainingSetIndices, shuffle=True, dataAugment=True, reSample=True)):
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
                if weight == 0:
                    continue

                if i ==0:
                    if epoch >= pivotEpoch:   #only train treatment reponse branch after epoch 1000.
                        outputs = xr
                        gt1, gt2 = (response1, response2)
                    else:
                        continue
                else:
                    # only train seg path before pivotEpoch
                    if epoch >=pivotEpoch:
                         continue
                    else:
                         outputs = xup
                         gt1, gt2 = (seg1, seg2)

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
            lrScheduler.step(trainingLoss)

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

        with torch.no_grad():
            for inputs, segCpu, responseCpu in dataMgr.dataSegResponseGenerator(dataMgr.m_validationSetIndices, shuffle=False, dataAugment=False, reSample=False):
                inputs, seg, response = torch.from_numpy(inputs), torch.from_numpy(segCpu), torch.from_numpy(responseCpu)
                inputs, seg, response = inputs.to(device, dtype=torch.float), seg.to(device, dtype=torch.long), response.to(device, dtype=torch.long)  # return a copy

                xr, xup = net.forward(inputs)
                loss = torch.tensor(0.0).to(device)

                for i, (lossFunc, weight) in enumerate(zip(net.module.m_lossFuncList if useDataParallel else net.m_lossFuncList,
                                                           lossWeightList)):
                    if i == 0:
                        if epoch >= pivotEpoch:  # only train treatment reponse branch after epoch 1000.
                            outputs = xr
                            gt = response
                        else:
                            continue
                    else:
                        # only train seg path before pivotEpoch
                        if epoch >= pivotEpoch:
                            continue
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

            if epoch % 5 == 0:
                responseTestAccuracy = dataMgr.getAccuracy(epochPredict, epochResponse)
                responseTestTPR = dataMgr.getTPR(epochPredict, epochResponse)[0]
                responseTestTNR = dataMgr.getTNR(epochPredict, epochResponse)[0]

        testDiceAvgList = [x / (y + 1e-8) for x, y in zip(testDiceSumList, testDiceCountList)]
        testTPRAvgList  = [x / (y + 1e-8) for x, y in zip(testTPRSumList, testTPRCountList)]

        outputString =  f'{epoch}\t{trainingLoss:.4f}\t' + f'\t'.join((f'{x:.3f}' for x in trainDiceAvgList)) + f'\t' + f'\t'.join((f'{x:.3f}' for x in trainTPRAvgList)) + f'\t{responseTrainAccuracy:.4f}' + f'\t{responseTrainTPR:.4f}' + f'\t{responseTrainTNR:.4f}'
        outputString += f'\t\t{testLoss:.4f}\t' + f'\t'.join((f'{x:.3f}' for x in testDiceAvgList)) + f'\t' + f'\t'.join((f'{x:.3f}' for x in testTPRAvgList)) + f'\t{responseTestAccuracy:.4f}'+f'\t{responseTestTPR:.4f}'+ f'\t{responseTestTNR:.4f}'
        logging.info(outputString)

        # =============save net parameters==============
        if trainingLoss < float('inf') and not math.isnan(trainingLoss):
            netMgr.saveNet()
            if responseTestAccuracy > bestTestPerf or (responseTestAccuracy == bestTestPerf and testLoss < oldTestLoss):
                oldTestLoss = testLoss
                bestTestPerf = responseTestAccuracy
                netMgr.saveBest(bestTestPerf)
            if 1.0 == responseTrainAccuracy:
                logging.info(f"\n\nresponse Train Accuracy == 1, Program exit.")
                break
        else:
            logging.info(f"\n\nError: training loss is infinity. Program exit.")
            break

    torch.cuda.empty_cache()
    logging.info(f"\n\n=============END of Training of SkyWatcher Predict Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    logging.info(f'\nProgram Ending Time: {str(curTime)}')


if __name__ == "__main__":
    main()
