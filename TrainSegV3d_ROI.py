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
from CustomizedLoss import *
from ConsistencyLoss import *

logNotes = r'''
Major program changes: 
      1  3D V model for primary cancer ROI;
      2  Uniform ROI size: 51*171*171 in z,y,x directon;
      3  Total 36 patient data, in which training data 24 patients, validation 6 patients, and test 6 patients;
      4  all 36 patients data have 50-80% 3D label;
      5  Dice coefficient is 3D dice coefficient against corresponding 3D ground truth;
      6  training data augmentation in the fly: affine in XY plane, translation in Z direction;
      7  In the bottle neck of V model, the latent vector has size of 512*2*9*9;
      Sep 16th, 2019:
      1   add dynamic loss weight according trainin  data;
      2   refine learning rate decay.
      Sep 21st, 2019
      1   add improved Boundary Loss2, and inherit the previous learningrate of network of pure CELoss;
      Sep 23rd, 2019:
      1   improve mean of boundary loss limited on the A,B regions;
      2   use log(segProb) instead of segProb in the boudary loss;
      3   CrossEntropy weight reduces 0.01 per 5 epochs from 1 to 0.01, while boundary Loss weight increase 0.01 per 5 epochs from 0.01 to 1. 
      Sep 24th, 2019
      1   Use boundaryLoss1, which is considering the whole volume. 
      Sep 25th, 2019
      1   use boundaryLoss3, which is a stronger gradient signal to improve loss.
      2   unbalanced weight for class is applied on logP,and just use boundaryLoss3 with CELoss.
      3   use CELoss and boundaryLoss together.
      4   Use truncated DistanceCrossEntropy Loss alone;
      5   change LRScheduler into reduce into Plateau with initial LR=0.1
      Sep 26th, 2019
      1   Add one layer in the bottom of V model;
      2   Add residual connnection in each layer;
      Sep 30th, 2019
      1   With size-reduced ROI of size 51*149*149;
      2   reduce the translation of data augmentation;
      3   reduce all data into 35 patients, excluding a very blur patient.
      Oct 5th, 2019
      1   use uniform physical size 147mm*147mm*147mm, input pixel size: 49*147*147 with spacing size 3mm*1mm*1mm;
      2   change V model with inputsize 49*147*147
      Oct 6th, 2019
      1   add filter number to 48 at the first layer. 
      Oct 7th, 2019
      1   restore to 32 of number of filters in the first layer;
      2   add bottom number of filters to 1024, and keep down sample and add filter number together. 
      Oct 8th, 2019
      1   discard the cancer with size exceeding 147mm*147mm*147mm; Now remains 29 patients data; 
      Oct 9th, 2019
      1   In the first layer of V model, remove the residual link; 
           with the residula link at first layer: Tr dice:54%, Validation Dice 27%, Test Dice 56%;  Not good.
      2   the final output layer, change into 1*1*1 convolution, instead of 3*3*3 convolution;
      3   add labelConsistencyLoss, it use 64 dimension feature extracted from 2 ends of V model:
           It gets stable Training Dice 61%, validation Dice 27%, and test dice 49%, for fold 0 in the fixed physical size:147mm*147mm*147mm; 
      Oct 11th, 2019
      1   use feature tensor just from the output end of V model. It is 32 dimensions.
          It gets stable Training Dice 61%, validation Dice 23%, and test dice 49%, for fold 0 in the fixed physical size:147mm*147mm*147mm; 
      2   windows size for consistency loss changes to 3;
      Oct 12th, 2019
      1   change image window level to 100/50; relaunch training;
      2   change consistencyLoss to use ground truth for comparing diff of feature vector;
      Oct 13th, 2019
      1    use conistencyLoss3: ((G1-G2)-(P1-P2))**2 as loss.
      
      Oct 18th, 2019
      1   use 48 filters at the first layer with inputsize 49*147*147 with scaled ROI.
      
      Oct 20th, 2019
      1   at final output layer of V model, change 1*1*1 conv to 5*5*5 conv, in order to consider context for final output
      
      Oct 23th, 2019
      1   change to MaxPool with 2*2*2 with stride 2;
        
      
       
      

          
         

Discarded changes:                  
          '''


def printUsage(argv):
    print("============Train Seg 3D  VNet for ROI around primary Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <scratch> <fullPathOfData>  <fullPathOfLabel> <k>  <GPUID_List> <ConsistencyLoss>")
    print("where: \n"
          "       scratch =0: continue to train basing on previous training parameters; scratch=1, training from scratch.\n"
          "       k=[0, K), the k-th fold in the K-fold cross validation.\n"
          "       GPUIDList: 0,1,2,3, the specific GPU ID List, separated by comma\n"
          "       ConsistencyLoss: 0 or 1")

def main():
    if len(sys.argv) != 8:
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
    useConsistencyLoss = bool(int(sys.argv[7]))
    if useConsistencyLoss:
        searchWindow = 7
    else:
        searchWindow = 0

    # addBoundaryLoss = True

    # ===========debug==================
    oneSampleTraining = False  # for debug
    useDataParallel = True if len(GPUIDList) > 1 else False  # for debug
    # ===========debug==================

    print(f'Program ID:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'.........')

    inputSuffix = ".npy"
    K_fold = 6
    batchSize = 1 * len(GPUIDList)

    print(f"batchSize = {batchSize}")
    numWorkers = 0

    device = torch.device(f"cuda:{GPUIDList[0]}" if torch.cuda.is_available() else "cpu")

    if scratch > 0:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
    else:
        timeStr = getStemName(netPath)

    if '/home/hxie1/' in netPath:
        logFile = f'/home/hxie1/Projects/OvarianCancer/trainLog/log_CV{k:d}_Consis{useConsistencyLoss:d}_{timeStr}.txt'
        isArgon = False
    elif '/Users/hxie1/' in netPath:
        logFile = f'/Users/hxie1/Projects/OvarianCancer/trainLog/log_CV{k:d}_Consis{useConsistencyLoss:d}_{timeStr}.txt'
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
        logging.info(f"Info: useConsistencyLoss = {useConsistencyLoss} and searchWindowSize= {searchWindow}\n")
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

    net = SegV3DModel(useConsistencyLoss=useConsistencyLoss)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0)
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.setOptimizer(optimizer)

    lossWeight = dataPartitions.getLossWeight()
    ceLoss = DistanceCrossEntropyLoss(weight=lossWeight) # or weight=torch.tensor([1.0, 8.7135]) for whole dataset
    net.appendLossFunc(ceLoss, 1)
    # boundaryLoss = BoundaryLoss1(weight=lossWeight)
    # net.appendLossFunc(boundaryLoss, 0)

    if useConsistencyLoss:
        net.m_consistencyLoss = ConsistencyLoss3(lambdaCoeff=1, windowSize=searchWindow)

    # Load network
    netMgr = NetMgr(net, netPath, device)

    bestTestPerf = 0
    if 2 == len(getFilesList(netPath, ".pt")):
        netMgr.loadNet("train")
        bestTestPerf = netMgr.loadBestTestPerf()
    else:
        logging.info(net.getParametersScale())

    # lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    mileStones = [50, 100, 200, 350, 500]
    '''
    if addBoundaryLoss:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1
        newMileStones = [lastEpoch + int(x) for x in mileStones]
        lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=newMileStones, gamma=0.1, last_epoch=lastEpoch)
    else:
        lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mileStones, gamma=0.1, last_epoch=lastEpoch)
    '''
    # lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=mileStones, gamma=0.1, last_epoch=lastEpoch)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)


    if useDataParallel:
        net = nn.DataParallel(net, device_ids=GPUIDList, output_device=device)

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

        # update loss weight list

        lossWeightList = net.module.getLossWeightList() if useDataParallel else net.getLossWeightList()
        if len(lossWeightList) > 1 and 100 <= epoch <= 600 and (epoch - 100) % 5 == 0:
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

            outputs, loss = net.forward(inputs, gts)
            loss = loss.sum()  # gather loss on different GPUs.
            optimizer.zero_grad()
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

                outputs, loss = net.forward(inputs,gts)
                loss = loss.sum()  # gather loss on different GPUs.
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
            lrScheduler.step(validationLoss)
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

                outputs, loss = net.forward(inputs, gts)
                loss = loss.sum()  # gather loss on different GPUs.
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

        # =============save net parameters before output txt==============
        if trainingLoss < float('inf') and not math.isnan(trainingLoss):
            netMgr.saveNet()
            if epoch >= 1000 and \
                    (validationDice > bestTestPerf or (
                            validationDice == bestTestPerf and validationLoss < oldTestLoss)):
                oldTestLoss = validationLoss
                bestTestPerf = validationDice
                netMgr.saveBest(bestTestPerf)

        # ===========print train and test progress===============
        learningRate = net.module.getLR() if useDataParallel else net.getLR()
        outputString = f'{epoch}' + f'\t{learningRate:1.4e}'
        outputString += f'\t\t{trainingLoss:.4f}' + f'\t\t{trainingDice:.5f}'
        outputString += f'\t\t{validationLoss:.4f}' + f'\t\t{validationDice:.5f}'
        outputString += f'\t\t{testLoss:.4f}' + f'\t\t{testDice:.5f}'
        logging.info(outputString)



    torch.cuda.empty_cache()
    logging.info(f"\n\n=============END of Training of ResNeXt V Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    logging.info(f'\nProgram Ending Time: {str(curTime)}')

if __name__ == "__main__":
    main()
