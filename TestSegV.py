import sys
import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim

torchSummaryPath = "/home/hxie1/Projects/pytorch-summary/torchsummary"
sys.path.append(torchSummaryPath)
from torchsummary import summary

from DataMgr import DataMgr
from SegVModel import SegVModel
from NetMgr  import NetMgr

def printUsage(argv):
    print("============Test Ovarian Cancer Segmentation V model=============")
    print("read all test files, and output their segmentation results.")
    print("Usage:")
    print(argv[0], "<netSavedPath> <fullPathOfTestImages>  <fullPathOfTestLabels>")

def main():
    curTime = datetime.datetime.now()
    print('\nProgram starting Time: ', str(curTime))

    if len(sys.argv) != 4:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    testDataMgr = DataMgr(sys.argv[2], sys.argv[3])
    testDataMgr.setDataSize(32, 21, 281, 281, 4)  # batchSize, depth, height, width, k
    testDataMgr.buildImageAttrList()

    net= SegVModel()
    net.printParametersScale()

    lossFunc = nn.CrossEntropyLoss()
    net.setLossFunc(lossFunc)

    netMgr = NetMgr(net, netPath)
    if 2 == len(testDataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet(False)  # False for test
    else:
        print(f"Program can not find trained network in path: {netPath}")
        sys.exit()

    useDataParallel = True

    # print model
    print("\n====================Net Architecture===========================")
    summary(net.cuda(), testDataMgr.getInputSize())
    print("===================End of Net Architecture =====================\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU >1:
            print(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net)
    net.to(device)

    K = testDataMgr.getNumClassification()
    print("Hints: TestDice_0 is the dice coeff for all non-zero labels")
    print("Hints: TestDice_1 is for primary cancer(green), testDice_2 is for metastasis(yellow), and testDice_3 is for invaded lymph node(brown).\n")
    diceHead = (f'TestDice_{i}' for i in range(K))
    print(f"Epoch \t TrainingLoss \t TestLoss \t\t", '\t\t'.join(diceHead))   # print output head

    net.eval()
    n = 0 # n indicate the first slice index in the dataMgr.m_segSliceTupleList
    with torch.no_grad():
        diceSumList = [0 for _ in range(K)]
        diceCountList = [0 for _ in range(K)]
        testLoss = 0.0
        batches = 0
        for inputsCpu, labelsCpu in testDataMgr.dataLabelGenerator(False):
            inputs, labels = torch.from_numpy(inputsCpu), torch.from_numpy(labelsCpu)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)  # return a copy

            outputs = net.forward(inputs)
            loss = lossFunc(outputs, labels)
            batchLoss = loss.item()

            outputs = outputs.cpu().numpy()
            segmentations = testDataMgr.oneHotArray2Segmentation(outputs)
            testDataMgr.saveInputsSegmentations2Images(inputsCpu, labelsCpu, segmentations, n)

            (diceSumBatch, diceCountBatch) = testDataMgr.getDiceSumList(segmentations, labelsCpu)
            diceSumList = [x + y for x, y in zip(diceSumList, diceSumBatch)]
            diceCountList = [x + y for x, y in zip(diceCountList, diceCountBatch)]

            testLoss += batchLoss
            batches += 1
            n += inputsCpu.shape[0]  # for dynamic batchSize
            #print(f'batch={batches}: batchLoss = {batchLoss}')

    #===========print train and test progress===============
    testLoss /= batches
    diceAvgList = [x / (y + 1e-8) for x, y in zip(diceSumList, diceCountList)]
    print(f'{0} \t\t {0:.7f} \t\t {testLoss:.7f} \t\t', '\t\t\t'.join( (f'{x:.4f}' for x in diceAvgList)))

    print(f'\nTotal test {n} images in {sys.argv[2]}.')

    torch.cuda.empty_cache()
    print("=============END of Test of Ovarian Cancer Segmentation V Model =================")

if __name__ == "__main__":
    main()