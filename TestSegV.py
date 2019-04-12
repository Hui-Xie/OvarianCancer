import sys
import datetime
import torch
import torch.nn as nn

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

    ceWeight = torch.FloatTensor([1, 39, 68, 30653])
    lossFunc = nn.CrossEntropyLoss(weight=ceWeight)
    #lossFunc = nn.CrossEntropyLoss()
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
    print("Hints: Test Dice_0 is the dice coeff for all non-zero labels")
    print("Hints: Test Dice_1 is for primary cancer(green), test Dice_2 is for metastasis(yellow), and test Dice_3 is for invaded lymph node(brown).")
    print("Hints: Test TPR_0 is the TPR for all non-zero labels")
    print("Hints: Test TPR_1 is for primary cancer(green), TPR_2 is for metastasis(yellow), and TPR_3 is for invaded lymph node(brown).\n")
    diceHead = (f'Dice_{i}' for i in range(K))
    TPRHead = (f'TPR_{i}' for i in range(K))
    print(f"Epoch \t TrainingLoss \t TestLoss \t", '\t'.join(diceHead),'\t', '\t'.join(TPRHead))  # print output head

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
            loss = lossFunc(outputs, labels)
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
    testLoss /= batches
    diceAvgList = [x / (y + 1e-8) for x, y in zip(diceSumList, diceCountList)]
    TPRAvgList = [x / (y + 1e-8) for x, y in zip(TPRSumList, TPRCountList)]
    print(f'{0} \t {0:.4f} \t {testLoss:.4f} \t', '\t'.join((f'{x:.3f}' for x in diceAvgList)),'\t', '\t'.join((f'{x:.3f}' for x in TPRAvgList)))

    print(f'\nTotal test {n} images in {sys.argv[2]}.')

    torch.cuda.empty_cache()
    print("=============END of Test of Ovarian Cancer Segmentation V Model =================")

if __name__ == "__main__":
    main()