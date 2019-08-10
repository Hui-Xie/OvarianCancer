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
            1   Use slices as features channels in convolutions,  and use 1*1 convolution along slices direction to implement z direction convolution followed by 3*3 convolutino inside slice planes;
                It just uses three cascading 2D convolutions (first z, then xy, and z direction again) to implement 3D convolution, like in the paper of ResNeXt below.
                The benefits of this design:
                A   reduce network parameters, hoping to reducing overfitting;
                B   speed up training;
                C   this implemented 3D convolutions are all in full slices space;
            2   use group convolution to implement thick slice convolution to increase the network representation capability;
            3   Use ResNeXt-based module like Paper "Aggregated Residual Transformations for Deep Neural Networks " 
                (Link: http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html);
            4   use rich 2D affine transforms slice by slice and concatenate them to implement 3D data augmentation;
            5   20% data for independent test, remaining 80% data for 4-fold cross validation;
            6   add lossweight to adjust positive samples to 3/7 posweight in BCEWithLogitsLoss;
            
            Update:
            1    reduced network parameters to 3.14 million in July 27th, 2019, 0840am
            2    at 15:00 of July 27th, 2019, reduce network parameter again. Now each stage has 160 filters, with 1.235 million parameters
            3    keep 2) parameter, change all maxpooling into average pooling.
            4    At July 29th 09:37am, 2019, reduce filters to 96 to further reduce parameters, keep avgPool.
            5    at July 29th 11:25am, 2019,  reduce filter number to 48, and redue one stage
            6    at July 29th 12:41, 2019:
                    add GPUID in command line;
                    use SGD optimizer, instead of Adam
                    add numbers of filters along deeper layer with step 12.
                    add saveDir's tims stamp;
            7    at July 29th 15:18, 2019,
                    change learning rate step_size = 5 from 10;
                    before FC, we use conv2d
                    learning rate start at 0.5.
            8    at July 30th 03:00, 2019:
                    add learning rate print;
                    use convStride =2;
                    add filter number by 2 times along deeper layers.
            9    at July 30th, 10:13, 2019:
                    add MaxPool2d in stage1;
                    add final filters to 2048.
            10   at July 30th, 15:23, 2019
                    final conv layer filter number: 1024
            11   at Aug 10th, 2019:
                    A. Add new patient data; and exclude non-standard patient data;
                    B. test the k-th fold,  validation on the (k+1)th fold;
                    C. new inputsize: 231*251*251 with pixels size 3*2*2 mm
                    D. data normalize into [0,1] after window level shresthold [0,300]
                    E. put data padding in to converting from nrrd to numpy;
                    
                                                        
                    
            
            
Discarded changes:                  
                  

Experiment setting:
Input CT data: maximum size 140*251*251 (zyx) of 3D numpy array with spacing size(5*2*2)
Ground truth: response binary label

Predictive Model: 

response Loss Function:  BCELogitLoss

Data:   total 220 patients, 5-fold cross validation, test 45, validation 45, and training 130.  

Training strategy: 

          '''


def printUsage(argv):
    print("============Train ResAttentionNet for Ovarian Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <fullPathOfData>  <fullPathOfResponseFile> k  GPUID")
    print("where: k=0-3, the k-th fold in the 4-fold cross validation.\n"
          "       GPUID=0-3, the specific GPU ID for single GPU running.\n")

def main():
    if len(sys.argv) != 6:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    dataInputsPath = sys.argv[2]
    responsePath = sys.argv[3]
    k = int(sys.argv[4])
    GPU_ID = int(sys.argv[5])  # choices: 0,1,2,3 for lab server.
    inputSuffix = ".npy"

    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
    trainLogFile = f'/home/hxie1/Projects/OvarianCancer/trainLog/log_ResAttention_CV{k:d}_{timeStr}.txt'
    logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')

    netPath = os.path.join(netPath, timeStr)
    print(f"=============training from sratch============")
    logging.info(f"=============training from sratch============")

    print(f'Program ID of Predictive Network training:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'Training log is in {trainLogFile}')
    print(f'.........')

    logging.info(f'Program ID: {os.getpid()}\n')
    logging.info(f'Program command: \n {sys.argv}')
    logging.info(logNotes)

    logging.info(f'\nProgram starting Time: {str(curTime)}')
    logging.info(f"Info: netPath = {netPath}\n")

    K_fold = 5
    logging.info(f"Info: this is the {k}th fold leave for test in the {K_fold}-fold cross-validation.\n")
    dataPartitions = OVDataPartition(dataInputsPath, responsePath, inputSuffix, K_fold, k, logInfoFun=logging.info)

    testTransform = OCDataTransform(0)
    trainTransform = OCDataTransform(0.9)
    validationTransform = OCDataTransform(0)

    testData = OVDataSet('test', dataPartitions, transform=testTransform, logInfoFun=logging.info)
    trainingData = OVDataSet('training', dataPartitions,  transform=trainTransform, logInfoFun=logging.info)
    validationData = OVDataSet('validation', dataPartitions,  transform=validationTransform, logInfoFun=logging.info)

    # ===========debug==================
    oneSampleTraining = False  # for debug
    useDataParallel = False  # for debug
    # ===========debug==================

    batchSize = 12  # 12 is for 1 GPU
    numWorkers = batchSize

    net = ResAttentionNet()
    # optimizer = optim.Adam(net.parameters(), weight_decay=0)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    net.setOptimizer(optimizer)

    lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

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

    bceWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.35/0.65]).to(device, dtype=torch.float))  # for imbalance training data
    net.appendLossFunc(bceWithLogitsLoss, 1)

    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU > 1:
            device_ids = [0,1]
            logging.info(f'Info: program will use {len(device_ids)} GPUs.')
            net = nn.DataParallel(net, device_ids=device_ids, output_device=device)

    if useDataParallel:
        logging.info(net.module.lossFunctionsInfo())
    else:
        logging.info(net.lossFunctionsInfo())

    epochs = 1500000

    logging.info(f"\nHints: Optimal_Result = Yes = 1,  Optimal_Result = No = 0 \n")

    logging.info(f"Epoch" + f"\tLearningRate"\
                 + f"\t\tTrLoss" + f"\tAccura" + f"\tTPR_r" + f"\tTNR_r" \
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
        learningRate  = lrScheduler.get_lr()[0]
        outputString  = f'{epoch}' +f'\t{learningRate:1.4e}'
        outputString += f'\t\t{trainingLoss:.4f}'       + f'\t{responseTrainAccuracy:.4f}'      + f'\t{responseTrainTPR:.4f}'      + f'\t{responseTrainTNR:.4f}'
        outputString += f'\t\t{validationLoss:.4f}'   + f'\t{responseValidationAccuracy:.4f}' + f'\t{responseValidationTPR:.4f}' + f'\t{responseValidationTNR:.4f}'
        outputString += f'\t\t{testLoss:.4f}'         + f'\t{responseTestAccuracy:.4f}'       + f'\t{responseTestTPR:.4f}'       + f'\t{responseTestTNR:.4f}'
        logging.info(outputString)

        # =============save net parameters==============
        if trainingLoss < float('inf') and not math.isnan(trainingLoss):
            netMgr.saveNet()
            if responseValidationAccuracy > bestTestPerf or (responseValidationAccuracy == bestTestPerf and validationLoss < oldTestLoss):
                oldTestLoss = validationLoss
                bestTestPerf = responseValidationAccuracy
                netMgr.saveBest(bestTestPerf)
            if trainingLoss <= 0.02: # CrossEntropy use natural logarithm . -ln(0.98) = 0.0202. it means training accuracy  for each sample gets 98% above
                logging.info(f"\n\n training loss less than 0.02, Program exit.")
                break
        else:
            logging.info(f"\n\nError: training loss is infinity. Program exit.")
            break

    torch.cuda.empty_cache()
    logging.info(f"\n\n=============END of Training of ResAttentionNet Predict Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    logging.info(f'\nProgram Ending Time: {str(curTime)}')


if __name__ == "__main__":
    main()
