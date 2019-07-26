# train ResNeXt-based Attention Net

import sys
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import logging

from OCDataSet import *
from FilesUtilities import *
from MeasureUtilities import *
from ResAttentionNet  import ResAttentionNet
from OCDataTransform import *
from NetMgr import NetMgr

logNotes = r'''
Major program changes: 
            ResNeXt-based Attention Net: use 2D network to implement 3D convolution without losing 3D context information. 
            0   the input is a 3D full volume without any cropping; 
            1   use slices as features channels in convolution, and use 1*1 convolution along slices to implement z direction convolution followed by 3*3 convolutino slice planes;
                it just use three cascading 2D convolutions (frist z, then xy, and z directon again) to implement 3D convolution, like in the paper of ResNeXt below.
                The benefits of this design:
                A   reduce network parameters, hoping to reducing overfitting;
                B   speed up training;
                C   this implemented 3D convolutions are all in full slices space;
            2   use group convolution to implement thick slice convolution to increase the network representation capability;
            3   Use ResNeXt-based module like Paper "Aggregated Residual Transformations for Deep Neural Networks " 
                (Link: http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html);
            4   use rich 2D affine transforms slice by slice and concatenate them to implement 3D data augmentation;
            5   20% data for independent test, remaining 80% data for 4-folc cross validation;

Discarded changes:                  
                  

Experiment setting:
Input CT data: maximum size 140*251*251 (zyx) of 3D numpy array with spacing size(5*2*2)
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
    k = int(sys.argv[4])
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
    logging.info(f"Info: netPath = {netPath}\n")

    K_fold = 4
    testRate = 0.2
    logging.info(f"Info: this is the {k}th fold leave for test in the {K_fold}-fold cross-validation, with {testRate:.1%} of data for independent test.\n")
    dataPartitions = OVDataPartition(dataInputsPath, responsePath, inputSuffix, K_fold, testProportion=testRate, logInfoFun=logging.info)

    testTransform = OCDataTransform(140, 251, 251, 0)
    trainTransform = OCDataTransform(140, 251, 251, 0.9)
    validationTransform = OCDataTransform(140, 251, 251, 0)

    testData = OVDataSet(dataPartitions, 'test', k, transform=testTransform, logInfoFun=logging.info)
    trainingData = OVDataSet(dataPartitions, 'train', k, transform=trainTransform, logInfoFun=logging.info)
    validationData = OVDataSet(dataPartitions, 'validation', k, transform=validationTransform, logInfoFun=logging.info)

    # ===========debug==================
    oneSampleTraining = False  # for debug
    useDataParallel = False  # for debug
    GPU_ID = 0  # choices: 0,1,2,3 for lab server.
    # ===========debug==================

    batchSize = 7  # 6 is for 1 GPU
    numWorkers = batchSize

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
        logging.info(f"=== Network trains from scratch ====")

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

    oldTestLoss = 1000

    for epoch in range(0, epochs):
        random.seed()
        if useDataParallel:
            lossFunc = net.module.getOnlyLossFunc()
        else:
            lossFunc = net.getOnlyLossFunc()
        # ================Training===============
        net.train()
        trainingLoss = 0.0
        trainingBatches = 0

        epochPredict = None
        epochResponse = None
        responseTrainAccuracy = 0.0
        responseTrainTPR = 0.0
        responseTrainTNR = 0.0


        for inputs, responseCpu in data.DataLoader(trainingData, batch_size=batchSize, shuffle=True, num_workers=numWorkers):
            inputs = inputs.to(device, dtype=torch.float)
            gt = responseCpu.to(device, dtype=torch.float)

            optimizer.zero_grad()
            xr = net.forward(inputs)
            loss = lossFunc(xr, gt)

            loss.backward()
            optimizer.step()
            batchLoss = loss.item()

            # accumulate response and predict value
            if epoch % 5 == 0:
                batchPredict = (xr>= 0).cpu().detach().numpy().flatten()
                epochPredict = np.concatenate(
                    (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                batchGt = responseCpu.detach().numpy()
                epochResponse = np.concatenate(
                    (epochResponse, batchGt)) if epochResponse is not None else batchGt


            trainingLoss += batchLoss
            trainingBatches += 1

            if oneSampleTraining:
                break

        if 0 != trainingBatches:
            trainingLoss /= trainingBatches
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
            for inputs, responseCpu in data.DataLoader(validationData, batch_size=batchSize, shuffle=False, num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gt     = responseCpu.to(device, dtype=torch.float)  # return a copy

                xr = net.forward(inputs)
                loss = lossFunc(xr, gt)
                batchLoss = loss.item()

                # accumulate response and predict value

                batchPredict = (xr>= 0).cpu().detach().numpy().flatten()
                epochPredict = np.concatenate(
                    (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                batchGt = responseCpu.detach().numpy()
                epochResponse = np.concatenate(
                    (epochResponse, batchGt)) if epochResponse is not None else batchGt

                validationLoss += batchLoss
                validationBatches += 1

                if oneSampleTraining:
                    break


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
                for inputs, responseCpu in data.DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=numWorkers):
                    inputs = inputs.to(device, dtype=torch.float)
                    gt = responseCpu.to(device, dtype=torch.float)  # return a copy

                    xr = net.forward(inputs)
                    loss = lossFunc(xr, gt)

                    batchLoss = loss.item()

                    # accumulate response and predict value

                    batchPredict = (xr>= 0).cpu().detach().numpy().flatten()
                    epochPredict = np.concatenate(
                        (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                    batchGt = responseCpu.detach().numpy()
                    epochResponse = np.concatenate(
                        (epochResponse, batchGt)) if epochResponse is not None else batchGt

                    testLoss += batchLoss
                    testBatches += 1

                    if oneSampleTraining:
                        break

                if 0 != testBatches:
                    testLoss /= testBatches

                if epoch % 5 == 0:
                    responseTestAccuracy = getAccuracy(epochPredict, epochResponse)
                    responseTestTPR = getTPR(epochPredict, epochResponse)[0]
                    responseTestTNR = getTNR(epochPredict, epochResponse)[0]


        # ===========print train and test progress===============
        outputString = f'{epoch}\t{trainingLoss:.4f}' + f'\t{responseTrainAccuracy:.4f}' + f'\t{responseTrainTPR:.4f}' + f'\t{responseTrainTNR:.4f}'
        outputString += f'\t\t{validationLoss:.4f}'   + f'\t{responseValidationAccuracy:.4f}' + f'\t{responseValidationTPR:.4f}' + f'\t{responseValidationTNR:.4f}'
        outputString += f'\t\t{testLoss:.4f}'         + f'\t{responseTestAccuracy:.4f}' + f'\t{responseTestTPR:.4f}' + f'\t{responseTestTNR:.4f}'
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
