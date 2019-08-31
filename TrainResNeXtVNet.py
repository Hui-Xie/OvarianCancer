# train ResNeXt V net
import sys
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import logging

from OCDataSegSet import *
from FilesUtilities import *
from MeasureUtilities import *
from ResNeXtVNet import ResNeXtVNet
from OCDataTransform import *
from NetMgr import NetMgr

logNotes = r'''
Major program changes: 
     1  a V model with ResNeXt block: use z convolution, and then xy convolution, to implement 3D convolution.
     2  at ground truth, only check the segmented slices, about 3 slices per patient;
     3  the input is whole 3D volume, instead of ROI around a segmented slice;
     4  support input data augmentation: affine in xy plane, and translation in z direction;
     5  input Size: 231*251*251 with label, instead of previous SkyWatch Model of 29*140*140;
     6  treat all 1,2,3 labels as 1, in other words, do not differentiate primary, metastase, and nymph node;  
    

Discarded changes:                  

Experiment setting:
Input CT data: maximum size 231*251*251 (zyx) of 3D numpy array with spacing size(3*2*2)

Loss Function:  BCELogitLoss

Data:   total 143 patients with weak annotaton label, 5-fold cross validation, test 29, validation 29, and training 85.  

Training strategy: 

          '''


def printUsage(argv):
    print("============Train ResNeXt VNet for Ovarian Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <scratch> <fullPathOfData>  <fullPathOfLabel> k  GPUID_List")
    print("where: \n"
          "       scratch =0: continue to train basing on previous training parameters; scratch=1, training from scratch.\n"
          "       k=[0, K), the k-th fold in the K-fold cross validation.\n"
          "       GPUIDList: 0,1,2,3, the specific GPU ID List, separated by comma\n")

def main():
    if len(sys.argv) != 7:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    scratch = int(sys.argv[2])
    dataInputsPath = sys.argv[3]
    groundTruthPath = sys.argv[4]
    k = int(sys.argv[5])
    GPUIDList = sys.argv[6].split(',')  # choices: 0,1,2,3 for lab server.
    GPUIDList = [int(x) for x in GPUIDList]

    # ===========debug==================
    oneSampleTraining = False  # for debug
    useDataParallel = True if len(GPUIDList) > 1 else False  # for debug
    # ===========debug==================

    print(f'Program ID of Predictive Network training:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'.........')

    inputSuffix = ".npy"
    K_fold = 5
    batchSize = 4 * len(GPUIDList)
    print(f"batchSize = {batchSize}")
    numWorkers = 0

    device = torch.device(f"cuda:{GPUIDList[0]}" if torch.cuda.is_available() else "cpu")

    if scratch > 0:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
    else:
        timeStr = getStemName(netPath)

    if '/home/hxie1/' in netPath:
        trainLogFile = f'/home/hxie1/Projects/OvarianCancer/trainLog/log_CV{k:d}_{timeStr}.txt'
        isArgon = False
    elif '/Users/hxie1/' in netPath:
        trainLogFile = f'/Users/hxie1/Projects/OvarianCancer/trainLog/log_CV{k:d}_{timeStr}.txt'
        isArgon = True
    else:
        print("output net path should be full path.")
        return
    print(f'Training log is in {trainLogFile}')
    logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')

    if scratch > 0:
        lastEpoch = -1
        netPath = os.path.join(netPath, timeStr)
        print(f"=============training from sratch============")
        logging.info(f"=============training from sratch============")

        logging.info(f'Program ID: {os.getpid()}\n')
        logging.info(f'Program command: \n {sys.argv}')
        logging.info(logNotes)

        logging.info(f'\nProgram starting Time: {str(curTime)}')
        logging.info(f"Info: netPath = {netPath}\n")

        logging.info(f"Info: this is the {k}th fold leave for test in the {K_fold}-fold cross-validation.\n")
        logging.info(f"Info: batchSize = {batchSize}\n")
        logging.info(f'Net parameters is saved in  {netPath}.')

    else:
        lastLine = getFinalLine(trainLogFile)
        lastRow = getListFromLine(lastLine)
        lastEpoch = int(lastRow[0])
        print(f"=============Training inheritates previous training at {netPath} ============")

    dataPartitions = OVDataSegPartition(dataInputsPath, groundTruthPath, inputSuffix, K_fold, k,
                                     logInfoFun=logging.info if scratch > 0 else print)

    trainTransform = OCDataLabelTransform(0.6)
    validationTransform = OCDataLabelTransform(0)
    testTransform = OCDataLabelTransform(0)

    trainingData = OVDataSegSet('training', dataPartitions, transform=trainTransform,
                             logInfoFun=logging.info if scratch > 0 else print)
    validationData = OVDataSegSet('validation', dataPartitions, transform=validationTransform,
                               logInfoFun=logging.info if scratch > 0 else print)
    testData = OVDataSegSet('test', dataPartitions, transform=testTransform,
                         logInfoFun=logging.info if scratch > 0 else print)

    net = ResNeXtVNet()
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=0)
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    net.setOptimizer(optimizer)

    # In all pixels of 441 labeled slices, 96% were labeled as 0, other were labeled as 1,2,3.
    bceWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(0.96/0.4), reduction="sum")
    net.appendLossFunc(bceWithLogitsLoss, 1)

    # Load network
    netMgr = NetMgr(net, netPath, device)

    bestTestPerf = 0
    if 2 == len(getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")  # True for train
        bestTestPerf = netMgr.loadBestTestPerf()
    else:
        logging.info(net.getParametersScale())

    # lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150, 300, 1200], gamma=0.1, last_epoch=lastEpoch)

    if useDataParallel:
        net = nn.DataParallel(net, device_ids=GPUIDList, output_device=device)
        lossFunc = net.module.getOnlyLossFunc()
    else:
        lossFunc = net.getOnlyLossFunc()

    epochs = 15000000
    oldTestLoss = 100000

    if scratch > 0:
       logging.info(f"\n\n************** Table of Train Log **************")
       logging.info(f"Epoch" + f"\tLearningRate" \
                     + f"\t\tTrainingLoss" +   f"\tDice" \
                     + f"\t\tValidationLoss" + f"\tDice" \
                     + f"\t\tTestLoss" +       f"\tDice" )  # logging.info output head

    for epoch in range(lastEpoch + 1, epochs):
        random.seed()

        # ================Training===============
        net.train()
        nSlice = 0

        trainingLoss = 0.0
        trainingBatches = 0
        trainingDice = 0.0

        for inputs, labels in data.DataLoader(trainingData, batch_size=batchSize, shuffle=True, num_workers=numWorkers):
            inputs = inputs.to(device, dtype=torch.float)
            gts = labels.to(device, dtype=torch.float)
            gts = (gts > 0).float() # not discriminate all non-zero labels.

            optimizer.zero_grad()
            outputs = net.forward(inputs)

            loss = torch.tensor(0.0).to(device, dtype=torch.float)
            gtsShape = gts.shape
            for i in range(gtsShape[0]):
                output = outputs[i,]
                gt = gts[i,]
                nonzeroSlices = torch.nonzero(gt, as_tuple=True)[0]
                nonzeroSlices = torch.unique(nonzeroSlices, sorted=True)
                slices = nonzeroSlices.shape[0]
                nSlice += slices
                for sPos in range(slices):
                    s = nonzeroSlices[sPos]
                    loss += lossFunc(output[s,], gt[s,])
                    trainingDice += tensorDice(output[s,], gt[s,])

            loss.backward()
            optimizer.step()
            batchLoss = loss.item()

            trainingLoss += batchLoss
            trainingBatches += 1

            if oneSampleTraining:
                break
        trainingDice = trainingDice/nSlice


        if 0 != trainingBatches:
            trainingLoss /= trainingBatches
            lrScheduler.step()

        if epoch % 5 != 0:
            continue  # only epoch %5 ==0, run validation set.

        # ================Validation===============
        net.eval()
        nSlice = 0
        validationLoss = 0.0
        validationBatches = 0
        validationDice = 0.0

        with torch.no_grad():
            for inputs, labels in data.DataLoader(validationData, batch_size=batchSize, shuffle=False, num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gts = labels.to(device, dtype=torch.float)  # return a copy
                gts = (gts > 0).float()  # not discriminate all non-zero labels.

                outputs = net.forward(inputs)

                loss = torch.tensor(0.0).to(device, dtype=torch.float)
                gtsShape = gts.shape
                for i in range(gtsShape[0]):
                    output = outputs[i,]
                    gt = gts[i,]
                    nonzeroSlices = torch.nonzero(gt, as_tuple=True)[0]
                    nonzeroSlices = torch.unique(nonzeroSlices, sorted=True)
                    slices = nonzeroSlices.shape[0]
                    nSlice += slices
                    for sPos in range(slices):
                        s = nonzeroSlices[sPos]
                        loss += lossFunc(output[s,], gt[s,])
                        validationDice += tensorDice(output[s,], gt[s,])

                batchLoss = loss.item()
                validationLoss += batchLoss
                validationBatches += 1

                if oneSampleTraining:
                    break

            if 0 != validationBatches:
                validationLoss /= validationBatches
            validationDice = validationDice / nSlice


        # ================Independent Test===============
        net.eval()
        nSlice =0

        testLoss = 0.0
        testBatches = 0
        testDice = 0.0

        with torch.no_grad():
            for inputs, labels in data.DataLoader(testData, batch_size=batchSize, shuffle=False,num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gts = labels.to(device, dtype=torch.float)  # return a copy
                gts = (gts > 0).float()  # not discriminate all non-zero labels.

                outputs = net.forward(inputs)

                loss = torch.tensor(0.0).to(device, dtype=torch.float)
                gtsShape = gts.shape
                for i in range(gtsShape[0]):
                    output = outputs[i,]
                    gt = gts[i,]
                    nonzeroSlices = torch.nonzero(gt, as_tuple=True)[0]
                    nonzeroSlices = torch.unique(nonzeroSlices, sorted=True)
                    slices = nonzeroSlices.shape[0]
                    nSlice += slices
                    for sPos in range(slices):
                        s = nonzeroSlices[sPos]
                        loss += lossFunc(output[s,], gt[s,])
                        testDice += tensorDice(output[s,], gt[s,])

                batchLoss = loss.item()
                testLoss += batchLoss
                testBatches += 1

                if oneSampleTraining:
                    break

            if 0 != testBatches:
                testLoss /= testBatches
            testDice = testDice / nSlice

        # ===========print train and test progress===============
        learningRate = lrScheduler.get_lr()[0]
        outputString = f'{epoch}' + f'\t{learningRate:1.4e}'
        outputString += f'\t\t{trainingLoss:.4f}' + f'\t{trainingDice:.5f}'
        outputString += f'\t\t{validationLoss:.4f}' + f'\t{validationDice:.5f}'
        outputString += f'\t\t{testLoss:.4f}' + f'\t{testDice:.5f}'
        logging.info(outputString)

        # =============save net parameters==============
        if trainingLoss < float('inf') and not math.isnan(trainingLoss):
            netMgr.saveNet()
            if validationDice  > bestTestPerf or (validationDice == bestTestPerf and validationLoss < oldTestLoss):
                oldTestLoss = validationLoss
                bestTestPerf = validationDice
                netMgr.saveBest(bestTestPerf)
            if trainingLoss <= 10:
                logging.info(f"\n\n training loss less than 10, Program exit.")
                break
        else:
            logging.info(f"\n\nError: training loss is infinity. Program exit.")
            break

    torch.cuda.empty_cache()
    logging.info(f"\n\n=============END of Training of ResNeXt V Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    logging.info(f'\nProgram Ending Time: {str(curTime)}')

if __name__ == "__main__":
    main()
