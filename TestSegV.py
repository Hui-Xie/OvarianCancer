import sys
import os
import datetime
import torch
import torch.nn as nn
import logging

torchSummaryPath = "/home/hxie1/Projects/pytorch-summary/torchsummary"
sys.path.append(torchSummaryPath)
from torchsummary import summary

from SegDataMgr import SegDataMgr
from SegV3DModel import SegV3DModel
from SegV2DModel import SegV2DModel
from NetMgr  import NetMgr
from CustomizedLoss import FocalCELoss, BoundaryLoss


# you may need to change the file name and log Notes below for every training.
testLogFile = r'''/home/hxie1/Projects/OvarianCancer/trainLog/test_20190504.txt'''
logNotes = r'''
            major program changes: ....
            '''

logging.basicConfig(filename=testLogFile,filemode='a+',level=logging.INFO, format='%(message)s')


def printUsage(argv):
    print("============Test Ovarian Cancer Segmentation V model=============")
    print("read all test files, and output their segmentation results.")
    print("Usage:")
    print(argv[0], "<netSavedPath> <fullPathOfTrainImages>  <fullPathOfTrainLabels>  <2D|3D> <labelTuple>")
    print("eg. labelTuple: (0,1,2,3), or (0,1), (0,2)")

def main():
    if len(sys.argv) != 6:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    print(f'Program ID {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'log is in {testLogFile}')
    print(f'.........')

    logging.info(f'Program ID {os.getpid()}\n')

    curTime = datetime.datetime.now()
    logging.info(f'\nProgram starting Time: {str(curTime)}')

    netPath = sys.argv[1]
    imagesPath = sys.argv[2]
    labelsPath = sys.argv[3]
    is2DInput = True if sys.argv[4] == "2D" else False
    logging.info(f"Info: netPath = {netPath}\n")
    labelTuple = eval(sys.argv[5])
    K = len(labelTuple)

    testDataMgr = SegDataMgr(imagesPath, labelsPath, logInfoFun=logging.info)
    testDataMgr.setRemainedLabel(3, labelTuple)

    # ===========debug==================
    testDataMgr.setOneSampleTraining(False)  # for debug
    useDataParallel = True  # for debug
    # ===========debug==================

    testDataMgr.buildSegSliceTupleList()

    if is2DInput:
        logging.info(f"Info: program uses 2D input.")
        testDataMgr.setDataSize(8, 1, 281, 281, K, "TestData")  # batchSize, depth, height, width, k
        net = SegV2DModel(128, K)  # 128 is the number of filters in the first layer for primary cancer.
    else:
        print("Info: program uses 3D input.")
        testDataMgr.setDataSize(8, 21, 281, 281, K, "TestData")  # batchSize, depth, height, width, k
        net = SegV3DModel(K)

    testDataMgr.buildImageAttrList()

    logging.info(net.getParametersScale())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ceWeight = torch.FloatTensor(testDataMgr.getCEWeight()).to(device)
    focalLoss = FocalCELoss(weight=ceWeight)
    net.appendLossFunc(focalLoss, 1)
    boundaryLoss = BoundaryLoss(lambdaCoeff=0.001, k=K)
    net.appendLossFunc(boundaryLoss, 0)

    netMgr = NetMgr(net, netPath)
    if 2 == len(testDataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet("test")  # False for test
        logging.info(f'Program loads net from {netPath}.')
    else:
        logging.info(f"Program can not find trained network in path: {netPath}")
        sys.exit()

    # print model
    logging.info("\n====================Net Architecture===========================")
    stdoutBackup = sys.stdout
    with open(testLogFile, 'a+') as log:
        sys.stdout = log
        summary(net.cuda(), testDataMgr.getInputSize())
    sys.stdout = stdoutBackup
    logging.info("===================End of Net Architecture =====================\n")


    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU >1:
            logging.info(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net)
    net.to(device)

    K = testDataMgr.getNumClassification()
    logging.info("Hints: Test Dice_0 is the dice coeff for all non-zero labels")
    logging.info("Hints: Test Dice_1 is for primary cancer(green), \ntest Dice_2 is for metastasis(yellow), \nand test Dice_3 is for invaded lymph node(brown).")
    logging.info("Hints: Test TPR_0 is the TPR for all non-zero labels")
    logging.info("Hints: Test TPR_1 is for primary cancer(green), \nTPR_2 is for metastasis(yellow), \nand TPR_3 is for invaded lymph node(brown).\n")
    diceHead = (f'Dice_{i}' for i in labelTuple)
    TPRHead = (f'TPR_{i}' for i in labelTuple)
    logging.info(f"Epoch \t TrainingLoss \t TestLoss \t"+ f'\t'.join(diceHead) + f'\t'+  f'\t'.join(TPRHead))  # print output head

    net.eval()
    n = 0 # n indicate the first slice index in the dataMgr.m_segSliceTupleList
    with torch.no_grad():
        diceSumList = [0 for _ in range(K)]
        diceCountList = [0 for _ in range(K)]
        TPRSumList = [0 for _ in range(K)]
        TPRCountList = [0 for _ in range(K)]
        testLoss = 0.0
        batches = 0
        for inputsCpu, labelsCpu in testDataMgr.dataLabelGenerator(False):
            inputs, labels = torch.from_numpy(inputsCpu), torch.from_numpy(labelsCpu)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)  # return a copy

            outputs = net.forward(inputs)
            loss = torch.tensor(0.0)
            for lossFunc, weight in zip(net.module.m_lossFuncList, net.module.m_lossWeightList):
                if weight == 0:
                    continue
                loss += lossFunc(outputs, labels) * weight
            batchLoss = loss.item()

            outputs = outputs.cpu().numpy()
            segmentations = testDataMgr.oneHotArray2Segmentation(outputs)
            testDataMgr.saveInputsSegmentations2Images(inputsCpu, labelsCpu, segmentations, n)

            (diceSumBatch, diceCountBatch) = testDataMgr.getDiceSumList(segmentations, labelsCpu)
            (TPRSumBatch, TPRCountBatch) = testDataMgr.getTPRSumList(segmentations, labelsCpu)

            diceSumList = [x + y for x, y in zip(diceSumList, diceSumBatch)]
            diceCountList = [x + y for x, y in zip(diceCountList, diceCountBatch)]
            TPRSumList = [x + y for x, y in zip(TPRSumList, TPRSumBatch)]
            TPRCountList = [x + y for x, y in zip(TPRCountList, TPRCountBatch)]

            testLoss += batchLoss
            batches += 1
            n += inputsCpu.shape[0]  # for dynamic batchSize
            #print(f'batch={batches}: batchLoss = {batchLoss}')

    #===========print train and test progress===============
    if 0 != batches:
        testLoss /= batches
    diceAvgList = [x / (y + 1e-8) for x, y in zip(diceSumList, diceCountList)]
    TPRAvgList = [x / (y + 1e-8) for x, y in zip(TPRSumList, TPRCountList)]
    logging.info(f'{0} \t {0:.4f} \t {testLoss:.4f} \t'+ f'\t'.join((f'{x:.3f}' for x in diceAvgList))+ f'\t'+  f'\t'.join((f'{x:.3f}' for x in TPRAvgList)))

    logging.info(f'\nTotal test {n} images in {imagesPath}.')

    torch.cuda.empty_cache()
    logging.info("=============END of Test of Ovarian Cancer Segmentation V Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')

if __name__ == "__main__":
    main()