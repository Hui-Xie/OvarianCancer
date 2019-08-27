# train optimal surgical result, survival, and chemo result at same time.
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
from ResAttentionNet import ResAttentionNet
from OCDataTransform import *
from NetMgr import NetMgr

logNotes = r'''
Major program changes: 
           1   Triple label: sugrical result, chemo result, survival
           2   Print result only for optimal response, which is and operation of above 3 results;
                  



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
          "<netSavedPath> <scratch> <fullPathOfData>  <fullPathOfGroundTruthFile> k  GPUID_List")
    print("where: \n"
          "       scratch =0: continue to train basing on previous training parameters; scratch=1, training from scratch.\n"
          "       k=[0, K), the k-th fold in the K-fold cross validation.\n"
          "       GPUIDList: 0,1,2,3, the specific GPU ID List, separated by comma\n")


def printPartNetworkPara(epoch, net):  # only support non-parallel
    print(f"Epoch: {epoch}   =================")
    print("FC.bias = ", net.m_fc1.bias.data)
    print("STN5 bias = ", net.m_stn5.m_regression.bias.data)
    print("STN4 bias = ", net.m_stn4.m_regression.bias.data)
    print("gradient at FC.bias=", net.m_fc1.bias._grad)
    print("\n")


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
    inputSuffix = ".npy"

    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

    if '/home/hxie1/' in netPath:
        trainLogFile = f'/home/hxie1/Projects/OvarianCancer/trainLog/log_ResAttention_CV{k:d}_{timeStr}.txt'
        isArgon = False
    elif '/Users/hxie1/' in netPath:
        trainLogFile = f'/Users/hxie1/Projects/OvarianCancer/trainLog/log_ResAttention_CV{k:d}_{timeStr}.txt'
        isArgon = True
    else:
        print("output net path should be full path.")
        return

    logging.basicConfig(filename=trainLogFile, filemode='a+', level=logging.INFO, format='%(message)s')

    if scratch > 0:
        netPath = os.path.join(netPath, timeStr)
        print(f"=============training from sratch============")
        logging.info(f"=============training from sratch============")
    else:
        print(f"=============training inheritates previous training of {netPath} ============")
        logging.info(f"=============training inheritates previous training of {netPath} ============")

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
    dataPartitions = OVDataPartition(dataInputsPath, groundTruthPath, inputSuffix, K_fold, k, logInfoFun=logging.info)

    testTransform = OCDataTransform(0)
    trainTransform = OCDataTransform(0.9)
    validationTransform = OCDataTransform(0)

    trainingData = OVDataSet('training', dataPartitions, transform=trainTransform, logInfoFun=logging.info)
    validationData = OVDataSet('validation', dataPartitions, transform=validationTransform, logInfoFun=logging.info)
    testData = OVDataSet('test', dataPartitions, transform=testTransform, logInfoFun=logging.info)

    # ===========debug==================
    oneSampleTraining = False  # for debug
    useDataParallel = True if len(GPUIDList) > 1 else False  # for debug
    # ===========debug==================

    batchSize = 4 * len(GPUIDList)
    # for Regulare Conv:  3 is for 1 GPU, 6 for 2 GPU
    # For Deformable Conv: 4 is for 1 GPU, 8 for 2 GPUs.

    numWorkers = 0
    logging.info(f"Info: batchSize = {batchSize}\n")

    net = ResAttentionNet()
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    device = torch.device(f"cuda:{GPUIDList[0]}" if torch.cuda.is_available() else "cpu")
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=0.1, weight_decay=0)
    # optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    net.setOptimizer(optimizer)

    # lrScheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 150, 300, 1200], gamma=0.1)

    # Load network
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

    # for imbalance training data for BCEWithLogitsLoss
    if "patientResponseDict" in groundTruthPath:
        posWeight = torch.tensor([0.35 / 0.65]).to(device, dtype=torch.float)
        logging.info("This predicts optimal response.")
    elif "patientSurgicalResults" in groundTruthPath:
        posWeight = torch.tensor([0.23 / 0.77]).to(device, dtype=torch.float)
        logging.info("This predicts surgical results.")
    elif "patientTripleResults" in groundTruthPath:
        posWeight = torch.tensor([0.23/ 0.77, 0.235/0.765, 0.061/0.939]).to(device, dtype=torch.float)
        logging.info("This predicts surgical results, chemo result, survival at same time.")
    else:
        posWeight = 1.0
        logging.info("!!!!!!! Some thing wrong !!!!!")
        return

    bceWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=posWeight, reduction="sum")
    net.appendLossFunc(bceWithLogitsLoss, 1)

    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU > 1:
            logging.info(f'Info: program will use GPU {GPUIDList} from all {nGPU} GPUs.')
            net = nn.DataParallel(net, device_ids=GPUIDList, output_device=device)

    if useDataParallel:
        logging.info(net.module.lossFunctionsInfo())
    else:
        logging.info(net.lossFunctionsInfo())

    epochs = 1500000

    logging.info(f"\nHints: Optimal_Result = Yes = 1,  Optimal_Result = No = 0 \n")

    logging.info(f"Epoch" + f"\tLearningRate" \
                 + f"\t\tTrLoss" + f"\tAccura" + f"\tTPR_r" + f"\tTNR_r" \
                 + f"\t\tVaLoss" + f"\tAccura" + f"\tTPR_r" + f"\tTNR_r" \
                 + f"\t\tTeLoss" + f"\tAccura" + f"\tTPR_r" + f"\tTNR_r")  # logging.info output head

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

        for inputs, responseCpu in data.DataLoader(trainingData, batch_size=batchSize, shuffle=True,
                                                   num_workers=numWorkers):
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
                xr = torch.prod(xr, dim=1)
                batchPredict = (xr >= 0).cpu().detach().numpy().flatten()
                epochPredict = np.concatenate(
                    (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                batchGt = responseCpu.detach().numpy()
                batchGt = np.prod(batchGt, axis=1)
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

        # printPartNetworkPara(epoch, net)
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
            for inputs, responseCpu in data.DataLoader(validationData, batch_size=batchSize, shuffle=False,
                                                       num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gt = responseCpu.to(device, dtype=torch.float)  # return a copy

                xr = net.forward(inputs)
                loss = lossFunc(xr, gt)
                batchLoss = loss.item()

                # accumulate response and predict value
                xr = torch.prod(xr, dim=1)
                batchPredict = (xr >= 0).cpu().detach().numpy().flatten()
                epochPredict = np.concatenate(
                    (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                batchGt = responseCpu.detach().numpy()
                batchGt = np.prod(batchGt, axis=1)
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
                for inputs, responseCpu in data.DataLoader(testData, batch_size=batchSize, shuffle=False,
                                                           num_workers=numWorkers):
                    inputs = inputs.to(device, dtype=torch.float)
                    gt = responseCpu.to(device, dtype=torch.float)  # return a copy

                    xr = net.forward(inputs)
                    loss = lossFunc(xr, gt)

                    batchLoss = loss.item()

                    # accumulate response and predict value
                    xr = torch.prod(xr, dim=1)
                    batchPredict = (xr >= 0).cpu().detach().numpy().flatten()
                    epochPredict = np.concatenate(
                        (epochPredict, batchPredict)) if epochPredict is not None else batchPredict
                    batchGt = responseCpu.detach().numpy()
                    batchGt = np.prod(batchGt, axis=1)
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
        learningRate = lrScheduler.get_lr()[0]
        outputString = f'{epoch}' + f'\t{learningRate:1.4e}'
        outputString += f'\t\t{trainingLoss:.4f}' + f'\t{responseTrainAccuracy:.4f}' + f'\t{responseTrainTPR:.4f}' + f'\t{responseTrainTNR:.4f}'
        outputString += f'\t\t{validationLoss:.4f}' + f'\t{responseValidationAccuracy:.4f}' + f'\t{responseValidationTPR:.4f}' + f'\t{responseValidationTNR:.4f}'
        outputString += f'\t\t{testLoss:.4f}' + f'\t{responseTestAccuracy:.4f}' + f'\t{responseTestTPR:.4f}' + f'\t{responseTestTNR:.4f}'
        logging.info(outputString)

        # =============save net parameters==============
        if trainingLoss < float('inf') and not math.isnan(trainingLoss):
            netMgr.saveNet()
            if responseValidationAccuracy > bestTestPerf or (
                    responseValidationAccuracy == bestTestPerf and validationLoss < oldTestLoss):
                oldTestLoss = validationLoss
                bestTestPerf = responseValidationAccuracy
                netMgr.saveBest(bestTestPerf)
            if trainingLoss <= 0.02:  # CrossEntropy use natural logarithm . -ln(0.98) = 0.0202. it means training accuracy  for each sample gets 98% above
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
