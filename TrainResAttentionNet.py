# train ResNeXt-based Attention Net

import sys
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import logging
import os
import numpy as np
import math

from OCDataSet import *
from FilesUtilities import *
from MeasureUtilities import *
from ResAttentionNet  import ResAttentionNet
from OCDataTransform import *
from NetMgr import NetMgr

logNotes = r'''
Major program changes: 
                      ResNeXt-based Attention Net

Discarded changes:                  
                  

Experiment setting for Image3d ROI to response:
Input CT data: maximum size 140*251*251 (zyx) 3D numpy array with spacing size(5*2*2)
Ground truth: response binary label


Predictive Model: 

response Loss Function:  BCELogitLoss

Data:   training data has 169 patients 


Training strategy: 

          '''


def printUsage(argv):
    print("============Train ResAttentionNet for Ovarian Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <fullPathOfData>  <fullPathOfResponseFile> k ")
    print("where: k=0-3, the k-th fold in the 4-fold cross validation.")

def main():
    if len(sys.argv) != 5:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    dataInputsPath = sys.argv[2]
    responsePath = sys.argv[3]
    k = int(sys.argv[5])
    inputSuffix = ".npy"

    curTime = datetime.datetime.now()
    trainLogFile = f'/home/hxie1/Projects/OvarianCancer/trainLog/log_ResAttention_CV{k:d}_{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}.txt'
    logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')

    print(f'Program ID of Predictive Network training:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'Training log is in {trainLogFile}')
    print(f'.........')

    logging.info(f'Program ID: {os.getpid()}\n')
    logging.info(f'Program command: \n {sys.argv}')
    logging.info(logNotes)

    logging.info(f'\nProgram starting Time: {str(curTime)}')

    K_fold = 4
    logging.info(f"Info: this is the {k}th fold leave for test in the {K_fold}-fold cross-validation.\n")

    logging.info(f"Info: netPath = {netPath}\n")

    dataPartitions = OVDataPartition(dataInputsPath, responsePath, inputSuffix, K_fold, testProportion=0.2, logInfoFun=logging.info)

    testTransform = OCDataTransform(140, 251, 251, 0)
    trainTransform = OCDataTransform(140, 251, 251, 0.9)
    validationTransform = OCDataTransform(140, 251, 251, 0)

    testData = OVDataSet(dataPartitions, 'test', k, transform=testTransform, logInfoFun=logging.info)
    trainingData = OVDataSet(dataPartitions, 'train', k, transform=trainTransform, logInfoFun=logging.info)
    validationData = OVDataSet(dataPartitions, 'validation', k, transform=validationTransform, logInfoFun=logging.info)

    # ===========debug==================
    oneSampleTraining = True  # for debug
    useDataParallel = True  # for debug
    GPU_ID = 1  # choices: 0,1,2,3 for lab server.
    # ===========debug==================

    batchSize = 9  # 12 for use 4 GPUs

    net = ResAttentionNet()
    optimizer = optim.Adam(net.parameters(), weight_decay=0)
    net.setOptimizer(optimizer)

    lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    # Load network
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    net.to(device)
    netMgr = NetMgr(net, netPath, device)

    bestTestPerf = 0
    if 2 == len(getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")  # True for train
        logging.info(f'Program loads net from {netPath}.')
        bestTestPerf = netMgr.loadBestTestPerf()
        logging.info(f'Current best test performance: {bestTestPerf}')
    else:
        logging.info(f"Network trains from scratch.")

    logging.info(net.getParametersScale())

    net.appendLossFunc(nn.BCEWithLogitsLoss(), 1)

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

    epochs = 1500000

    logging.info(f"\nHints: Optimal_Result = Yes = 1,  Optimal_Result = No = 0 \n")

    logging.info(f"Epoch\tTrLoss" + f"\tAccura" + f"\tTPR_r" + f"\tTNR_r" \
                 + f"\t\tVaLoss" +  f"\tAccura" + f"\tTPR_r" + f"\tTNR_r" \
                 + f"\t\tTeLoss" +  f"\tAccura" + f"\tTPR_r" + f"\tTNR_r" )  # logging.info output head

    oldTrainingLoss = 1000
    oldTestLoss = 1000

    for epoch in range(0, epochs):
        random.seed()
        # ================Training===============
        net.train()
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

        for inputs, responseCpu in data.DataLoader(trainingData, batch_size=batchSize, shuffle=True, num_workers=9):
            inputs = inputs.to(device, dtype=torch.float)
            gt = responseCpu.to(device, dtype=torch.long)

            optimizer.zero_grad()
            xr = net.forward(inputs)
            loss = torch.tensor(0.0).to(device)

            for i, (lossFunc, weight) in enumerate(
                    zip(net.module.m_lossFuncList if useDataParallel else net.m_lossFuncList,
                        lossWeightList)):
                if weight == 0:
                    continue
                loss += lossFunc(xr, gt) * weight

            loss.backward()
            optimizer.step()
            batchLoss = loss.item()

            # accumulate response and predict value
            if epoch % 5 == 0:
                batchPredict = torch.argmax(xr, dim=1).cpu().detach().numpy().flatten()
                epochPredict = np.concatenate(
                    (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                epochResponse = np.concatenate(
                    (epochResponse, responseCpu)) if epochResponse is not None else responseCpu


            trainingLoss += batchLoss
            trainBatches += 1

        if 0 != trainBatches:
            trainingLoss /= trainBatches
            lrScheduler.step()

        if epoch % 5 == 0:
            responseTrainAccuracy = getAccuracy(epochPredict, epochResponse)
            responseTrainTPR = getTPR(epochPredict, epochResponse)[0]
            responseTrainTNR = getTNR(epochPredict, epochResponse)[0]
        else:
            continue  # only epoch %5 ==0, run validation set.

        # ================Validation===============
        net.eval()

        validationLoss = 0.0
        validationBatches = 0

        epochPredict = None
        epochResponse = None
        responseValidationAccuracy = 0.0
        responseValidationTPR = 0.0
        responseValidationTNR = 0.0

        with torch.no_grad():
            for inputs, responseCpu in data.DataLoader(validationData, batch_size=batchSize, shuffle=False, num_workers=9):
                inputs, gt = inputs.to(device, dtype=torch.float), responseCpu.to(device, dtype=torch.long)  # return a copy

                xr = net.forward(inputs)
                loss = torch.tensor(0.0).to(device)

                for i, (lossFunc, weight) in enumerate(
                        zip(net.module.m_lossFuncList if useDataParallel else net.m_lossFuncList,
                            lossWeightList)):
                    if weight == 0:
                        continue
                    loss += lossFunc(xr, gt) * weight

                batchLoss = loss.item()

                # accumulate response and predict value

                batchPredict = torch.argmax(xr, dim=1).cpu().detach().numpy().flatten()
                epochPredict = np.concatenate(
                    (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                epochResponse = np.concatenate(
                    (epochResponse, responseCpu)) if epochResponse is not None else responseCpu

                validationLoss += batchLoss
                validationBatches += 1


            if 0 != validationBatches:
                validationLoss /= validationBatches

            if epoch % 5 == 0:
                responseValidationAccuracy = getAccuracy(epochPredict, epochResponse)
                responseValidationTPR = getTPR(epochPredict, epochResponse)[0]
                responseValidationTNR = getTNR(epochPredict, epochResponse)[0]

            # ================Independent Test===============
            net.eval()

            testLoss = 0.0
            testBatches = 0

            epochPredict = None
            epochResponse = None
            responseTestAccuracy = 0.0
            responseTestTPR = 0.0
            responseTestTNR = 0.0

            with torch.no_grad():
                for inputs, responseCpu in data.DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=9):
                    inputs, gt = inputs.to(device, dtype=torch.float), responseCpu.to(device,  dtype=torch.long)  # return a copy

                    xr = net.forward(inputs)
                    loss = torch.tensor(0.0).to(device)

                    for i, (lossFunc, weight) in enumerate(
                            zip(net.module.m_lossFuncList if useDataParallel else net.m_lossFuncList,
                                lossWeightList)):
                        if weight == 0:
                            continue
                        loss += lossFunc(xr, gt) * weight

                    batchLoss = loss.item()

                    # accumulate response and predict value

                    batchPredict = torch.argmax(xr, dim=1).cpu().detach().numpy().flatten()
                    epochPredict = np.concatenate(
                        (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                    epochResponse = np.concatenate(
                        (epochResponse, responseCpu)) if epochResponse is not None else responseCpu

                    testLoss += batchLoss
                    testBatches += 1

                if 0 != testBatches:
                    testLoss /= testBatches

                if epoch % 5 == 0:
                    responseTestAccuracy = getAccuracy(epochPredict, epochResponse)
                    responseTestTPR = getTPR(epochPredict, epochResponse)[0]
                    responseTestTNR = getTNR(epochPredict, epochResponse)[0]


        # ===========print train and test progress===============
        outputString = f'{epoch}\t{trainingLoss:.4f}\t' + f'\t{responseTrainAccuracy:.4f}' + f'\t{responseTrainTPR:.4f}' + f'\t{responseTrainTNR:.4f}'
        outputString += f'\t\t{validationLoss:.4f}\t'   + f'\t{responseValidationAccuracy:.4f}' + f'\t{responseValidationTPR:.4f}' + f'\t{responseValidationTNR:.4f}'
        outputString += f'\t\t{testLoss:.4f}\t' + f'\t{responseTestAccuracy:.4f}' + f'\t{responseTestTPR:.4f}' + f'\t{responseTestTNR:.4f}'
        logging.info(outputString)

        # =============save net parameters==============
        if trainingLoss < float('inf') and not math.isnan(trainingLoss):
            netMgr.saveNet()
            if responseValidationAccuracy > bestTestPerf or (responseValidationAccuracy == bestTestPerf and validationLoss < oldTestLoss):
                oldTestLoss = validationLoss
                bestTestPerf = responseValidationAccuracy
                netMgr.saveBest(bestTestPerf)
            if 1.0 == responseTrainAccuracy:
                logging.info(f"\n\nresponse Train Accuracy == 1, Program exit.")
                break
        else:
            logging.info(f"\n\nError: training loss is infinity. Program exit.")
            break

    torch.cuda.empty_cache()
    logging.info(f"\n\n=============END of Training of RexAttentionNet Predict Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    logging.info(f'\nProgram Ending Time: {str(curTime)}')


if __name__ == "__main__":
    main()
