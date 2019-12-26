#  train Skywatcher Model for pure treatment response

import sys
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import numpy as np

from image2Predict.Image3dResponseDataMgr import Image3dResponseDataMgr
from OvarianCancerSeg.trial.SkyWatcherModel2 import SkyWatcherModel2
from framework.NetMgr import NetMgr
from framework.CustomizedLoss import FocalCELoss

# you may need to change the file name and log Notes below for every training.
trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/log_SkyWatcher_PurePrediction_20190624.txt'''
# trainLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/log_temp_20190610.txt'''
logNotes = r'''
Major program changes: 
                      merge train and test imageDataMgr into one.
                      when epoch %5 ==0, do not use mixup.
                      Directly use 3D data for treatment prediction without segmentation. 
                      Number of filters in the first layer in encoder is 32.
                      Only epoch %5 ==0, print log
                      Do not use normalization in FC layers.
                      Use cropping VOI on the fly.
                       

Experiment setting for Image3d ROI to response:
Input CT data: 29*140*140  3D CT raw image ROI with spacing size(5*2*2)

Predictive Model: 1,  first 3-layer dense conv block with channel size 128.
                  2,  and 3 dense conv DownBB blocks,  each of which includes a convStride 2 conv and 3-layers dense conv block; 
                  3,  and 3 fully connected layers  changes the tensor into size 2*1;
                  4,  final a softmax for binary classification;
                  Total network learning parameters are 8 million.
                  Network architecture is referred at https://github.com/Hui-Xie/OvarianCancer/blob/master/SkyWatcherModel.py

response Loss Function:   focus loss  with weight [3.3, 1.4] for [0,1] class separately, as [0,1] uneven distribution.

Data:   training data has 113 patients, and valdiation data has 27 patients with training/test rate 80/20.
        We randomize all data, and then assign same distrubtion of treat reponse 0,1 into to training and test data set.


Training strategy:  50% probability of data are mixed up with beta distribution with alpha =0.4, to feed into network for training. 
                    No other data augmentation, and no dropout.  

                    Learning Scheduler:  Reduce learning rate on  plateau, and learning rate patience is 30 epochs.                                

            '''

logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')


def printUsage(argv):
    print("============Train SkyWatcher Model for Ovarian Cancer  pure prediction=============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <fullPathOfData>  <fullPathOfResponseFile> ")


def main():
    if len(sys.argv) != 4:
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
    responsePath = sys.argv[3]
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
    dataMgr.setOneSampleTraining(False)  # for debug
    useDataParallel = True  # for debug
    GPU_ID = 1  # choices: 0,1,2,3 for lab server.
    # ===========debug==================

    batchSize = 9
    C = 32  # number of channels after the first input layer
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
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.

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

    # ========= end of loss function =================


    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU > 1:
            device_ids = [1, 2, 3]
            logging.info(f'Info: program will use {len(device_ids)} GPUs.')
            net = nn.DataParallel(net, device_ids=[1, 2, 3], output_device=device)

    if useDataParallel:
        logging.info(net.module.lossFunctionsInfo())
    else:
        logging.info(net.lossFunctionsInfo())

    epochs = 1500000

    logging.info(f"\nHints: Optimal_Result = Yes = 1,  Optimal_Result = No = 0 \n")

    logging.info(f"Epoch\tTrLoss" + f"\tAccura" + f"\tTPR_r"+ f"\tTNR_r"  + f"\t\tTsLoss" + f"\tAccura" + f"\tTPR_r" + f"\tTNR_r")  # logging.info output head

    oldTestLoss = 1000
    oldTrainingLoss = 1000

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

        if useDataParallel:
            lossWeightList = torch.Tensor(net.module.m_lossWeightList).to(device)
        else:
            lossWeightList = torch.Tensor(net.m_lossWeightList).to(device)

        for (inputs1, response1Cpu), (inputs2, response2Cpu) in zip(
                                            dataMgr.dataResponseGenerator(dataMgr.m_trainingSetIndices, shuffle=True, dataAugment=True, reSample=True),
                                            dataMgr.dataResponseGenerator(dataMgr.m_trainingSetIndices, shuffle=True, dataAugment=True, reSample=True)):
            if epoch % 5 == 0:
                lambdaInBeta = 1  # this will make the comparison in the segmention per 5 epochs meaningful.
            else:
                lambdaInBeta = dataMgr.getLambdaInBeta()

            inputs = inputs1 * lambdaInBeta + inputs2 * (1 - lambdaInBeta)
            inputs = torch.from_numpy(inputs).to(device, dtype=torch.float)
            response1 = torch.from_numpy(response1Cpu).to(device, dtype=torch.long)
            response2 = torch.from_numpy(response2Cpu).to(device, dtype=torch.long)


            optimizer.zero_grad()
            xr = net.forward(inputs, bPurePrediction=True)
            loss = torch.tensor(0.0).to(device)

            for (lossFunc, weight) in zip(net.module.m_lossFuncList if useDataParallel else net.m_lossFuncList,
                                          lossWeightList):
                outputs = xr
                gt1, gt2 = (response1, response2)

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
                batchPredict = torch.argmax(xr, dim=1).detach().cpu().numpy().flatten()
                epochPredict = np.concatenate((epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                epochResponse = np.concatenate((epochResponse, response1Cpu)) if epochResponse is not None else response1Cpu

            trainingLoss += batchLoss
            trainBatches += 1

        if 0 != trainBatches:
            trainingLoss /= trainBatches

        if epoch % 5 == 0:
            responseTrainAccuracy = dataMgr.getAccuracy(epochPredict, epochResponse)
            responseTrainTPR = dataMgr.getTPR(epochPredict, epochResponse)[0]
            responseTrainTNR = dataMgr.getTNR(epochPredict, epochResponse)[0]
        else:
            continue

        # ================Test===============
        net.eval()

        testLoss = 0.0
        testBatches = 0

        epochPredict = None
        epochResponse = None
        responseTestAccuracy = 0.0
        responseTestTPR = 0.0
        responseTestTNR = 0.0

        if not mergeTrainTestData:

            with torch.no_grad():
                for inputs, responseCpu in dataMgr.dataResponseGenerator(dataMgr.m_validationSetIndices, shuffle=True, dataAugment=False, reSample=False):
                    inputs, response = torch.from_numpy(inputs), torch.from_numpy(responseCpu)
                    inputs, response = inputs.to(device, dtype=torch.float), response.to(device, dtype=torch.long)  # return a copy

                    xr = net.forward(inputs, bPurePrediction=True)
                    loss = torch.tensor(0.0).to(device)
                    for (lossFunc, weight) in zip(net.module.m_lossFuncList if useDataParallel else net.m_lossFuncList,
                                                  lossWeightList):
                        outputs = xr
                        gt = response
                        if weight != 0:
                            loss += lossFunc(outputs, gt) * weight
                    batchLoss = loss.item()

                    # accumulate response and predict value
                    batchPredict = torch.argmax(xr, dim=1).detach().cpu().numpy().flatten()
                    epochPredict = np.concatenate((epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                    epochResponse = np.concatenate((epochResponse, responseCpu)) if epochResponse is not None else responseCpu

                    testLoss += batchLoss
                    testBatches += 1

                # ===========print train and test progress===============
                if 0 != testBatches:
                    testLoss /= testBatches
                    lrScheduler.step(testLoss)

                responseTestAccuracy = dataMgr.getAccuracy(epochPredict, epochResponse)
                responseTestTPR = dataMgr.getTPR(epochPredict, epochResponse)[0]
                responseTestTNR = dataMgr.getTNR(epochPredict, epochResponse)[0]


        else:
            lrScheduler.step(trainingLoss)

        outputString = f'{epoch}\t{trainingLoss:.4f}' + f'\t{responseTrainAccuracy:.4f}' + f'\t{responseTrainTPR:.4f}' + f'\t{responseTrainTNR:.4f}'
        outputString +=         f'\t\t{testLoss:.4f}' + f'\t{responseTestAccuracy:.4f}'  + f'\t{responseTestTPR:.4f}'  + f'\t{responseTestTNR:.4f}'
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
    logging.info(f"=============END of Training of SkyWatcher Pure Predict Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')


if __name__ == "__main__":
    main()
