# train Seg 3d V model
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
from SegV3DModel import SegV3DModel
from OCDataTransform import *
from NetMgr import NetMgr

logNotes = r'''
Major program changes: 
      1  3D V model for primary cancer ROI;
      2  Uniform ROI size: 51*171*171 in z,y,x directon;
      3  Total 36 patient data, in which training data 24 patients, validation 6 patients, and test 6 patients;
      4  all 36 patients data have 50-80% 3D label;
      5  Dice coefficient is 3D dice coefficient against corresponding 3D ground truth;
      6  training data augmentation in the fly: affine in XY plane, translation in Z direction;
      7  In the bottle neck of V model, the latent vector has size of 512*2*9*9;
         

Discarded changes:                  

Experiment setting:
Input CT data: 51*171*171 ROI around primary cancer

Loss Function:  SoftMax

Data:   total 36 patients with 50-80% label, 6-fold cross validation, test 6, validation 6, and training 24.  

Training strategy: 

          '''


def printUsage(argv):
    print("============Train Seg 3D  VNet for ROI around primary Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <scratch> <fullPathOfData>  <fullPathOfLabel> <k>  <GPUID_List>")
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

    print(f'Program ID:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'.........')

    inputSuffix = ".npy"
    K_fold = 6
    batchSize = 2 * len(GPUIDList)
    print(f"batchSize = {batchSize}")
    numWorkers = 0

    device = torch.device(f"cuda:{GPUIDList[0]}" if torch.cuda.is_available() else "cpu")

    if scratch > 0:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
    else:
        timeStr = getStemName(netPath)

    if '/home/hxie1/' in netPath:
        logFile = f'/home/hxie1/Projects/OvarianCancer/trainLog/log_CV{k:d}_{timeStr}.txt'
        isArgon = False
    elif '/Users/hxie1/' in netPath:
        logFile = f'/Users/hxie1/Projects/OvarianCancer/trainLog/log_CV{k:d}_{timeStr}.txt'
        isArgon = True
    else:
        print("the net path should be full path.")
        return
    print(f'Training log is in {logFile}')
    logging.basicConfig(filename=logFile, filemode='a+', level=logging.INFO, format='%(message)s')

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
        lastLine = getFinalLine(logFile)
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

    net = SegV3DModel()
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=0)
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    net.setOptimizer(optimizer)

    # In all pixels of 441 labeled slices, 96% were labeled as 0, other were labeled as 1,2,3.
    # image input size: 231*251*251, while its original avg size: 149*191*191 for weaked labeled nrrd; therefore, avg slice area increases 73%
    # so 0.96*1.73/0.04 is better pos_weight.
    # considering to that missing cancer has bigger risk cost, use 1.82 instead of 1.73 as factor. 
    ceLoss = nn.CrossEntropyLoss()
    net.appendLossFunc(ceLoss, 1)

    # Load network
    netMgr = NetMgr(net, netPath, device)

    bestTestPerf = 0
    if 2 == len(getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")
        bestTestPerf = netMgr.loadBestTestPerf()
    else:
        logging.info(net.getParametersScale())

    # lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150, 250, 400, 600, 800, 1000, 1200, 1400], gamma=0.1, last_epoch=lastEpoch)

    if useDataParallel:
        net = nn.DataParallel(net, device_ids=GPUIDList, output_device=device)
        lossFunc = net.module.getOnlyLossFunc()
    else:
        lossFunc = net.getOnlyLossFunc()

    epochs = 15000000
    oldTestLoss = 100000

    if scratch > 0:
       logging.info(f"\n\n************** Table of Training Log **************")
       logging.info(f"Epoch" + f"\tLearningRate" \
                     + f"\t\tTrainingLoss" +   f"\tDice" \
                     + f"\t\tValidationLoss" + f"\tDice" \
                     + f"\t\tTestLoss" +       f"\tDice" )  # logging.info output head

    for epoch in range(lastEpoch + 1, epochs):
        random.seed()

        # ================Training===============
        net.train()
        nSample = 0

        trainingLoss = 0.0
        trainingBatches = 0
        trainingDice = 0.0

        for inputs, labels, patientIDs in data.DataLoader(trainingData, batch_size=batchSize, shuffle=True, num_workers=numWorkers):
            inputs = inputs.to(device, dtype=torch.float)
            gts = labels.to(device, dtype=torch.float)
            gts = (gts > 0).long() # not discriminate all non-zero labels.

            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = lossFunc(outputs, gts)
            loss.backward()
            optimizer.step()
            batchLoss = loss.item()

            # compute dice
            with torch.no_grad():
                gtsShape = gts.shape
                for i in range(gtsShape[0]):
                    output = outputs[i,]
                    gt = gts[i,]
                    trainingDice += tensorDice(torch.argmax(output, dim=0), gt)
                nSample += gtsShape[0]

            trainingLoss += batchLoss
            trainingBatches += 1

            if oneSampleTraining:
                break
        trainingDice = trainingDice / nSample


        if 0 != trainingBatches:
            trainingLoss /= trainingBatches
            lrScheduler.step()

        if epoch % 5 != 0:
            continue  # only epoch %5 ==0, run validation set.

        # ================Validation===============
        net.eval()
        nSample = 0
        validationLoss = 0.0
        validationBatches = 0
        validationDice = 0.0

        with torch.no_grad():
            for inputs, labels, patientIDs in data.DataLoader(validationData, batch_size=batchSize, shuffle=False, num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gts = labels.to(device, dtype=torch.float)  # return a copy
                gts = (gts > 0).long()  # not discriminate all non-zero labels.

                outputs = net.forward(inputs)
                loss = lossFunc(outputs, gts)
                batchLoss = loss.item()

                # compute dice
                gtsShape = gts.shape
                for i in range(gtsShape[0]):
                    output = outputs[i,]
                    gt = gts[i,]
                    validationDice += tensorDice(torch.argmax(output, dim=0), gt)
                nSample += gtsShape[0]

                validationLoss += batchLoss
                validationBatches += 1

                if oneSampleTraining:
                    break

            if 0 != validationBatches:
                validationLoss /= validationBatches
            validationDice = validationDice / nSample


        # ================Independent Test===============
        net.eval()
        nSample =0

        testLoss = 0.0
        testBatches = 0
        testDice = 0.0

        with torch.no_grad():
            for inputs, labels, patientIDs in data.DataLoader(testData, batch_size=batchSize, shuffle=False,num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gts = labels.to(device, dtype=torch.float)  # return a copy
                gts = (gts > 0).long()  # not discriminate all non-zero labels.

                outputs = net.forward(inputs)
                loss = lossFunc(outputs, gts)
                batchLoss = loss.item()

                # compute dice
                gtsShape = gts.shape
                for i in range(gtsShape[0]):
                    output = outputs[i,]
                    gt = gts[i,]
                    testDice += tensorDice(torch.argmax(output, dim=0), gt)
                nSample += gtsShape[0]

                testLoss += batchLoss
                testBatches += 1

                if oneSampleTraining:
                    break

            if 0 != testBatches:
                testLoss /= testBatches
            testDice = testDice / nSample

        # ===========print train and test progress===============
        learningRate = lrScheduler.get_lr()[0]
        outputString = f'{epoch}' + f'\t{learningRate:1.4e}'
        outputString += f'\t\t{trainingLoss:.4f}' + f'\t\t{trainingDice:.5f}'
        outputString += f'\t\t{validationLoss:.4f}' + f'\t\t{validationDice:.5f}'
        outputString += f'\t\t{testLoss:.4f}' + f'\t\t{testDice:.5f}'
        logging.info(outputString)

        # =============save net parameters==============
        if trainingLoss < float('inf') and not math.isnan(trainingLoss):
            netMgr.saveNet()
            if validationDice  > bestTestPerf or (validationDice == bestTestPerf and validationLoss < oldTestLoss):
                oldTestLoss = validationLoss
                bestTestPerf = validationDice
                netMgr.saveBest(bestTestPerf)
            if trainingLoss <= 1e-6:
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
