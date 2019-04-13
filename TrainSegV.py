import sys
import datetime
import random
import torch
import torch.nn as nn
import torch.optim as optim

torchSummaryPath = "/home/hxie1/Projects/pytorch-summary/torchsummary"
sys.path.append(torchSummaryPath)
from torchsummary import summary

from DataMgr import DataMgr
from SegV3DModel import SegV3DModel
from SegV2DModel import SegV2DModel
from NetMgr  import NetMgr
from FocalCELoss import FocalCELoss



def printUsage(argv):
    print("============Train Ovarian Cancer Segmentation V model=============")
    print("Usage:")
    print(argv[0], "<netSavedPath> <fullPathOfTrainImages>  <fullPathOfTrainLabels>  <2D|3D>")

def main():
    curTime = datetime.datetime.now()
    print('\nProgram starting Time: ', str(curTime))

    if len(sys.argv) != 5:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    imagesPath = sys.argv[2]
    labelsPath = sys.argv[3]
    is2DInput = True if sys.argv[4] == "2D" else False
    print(f"Info: netPath = {netPath}\n")

    trainDataMgr = DataMgr(imagesPath, labelsPath)
    testDataMgr = DataMgr(*trainDataMgr.getTestDirs())

    if is2DInput:
        print("Info: program uses 2D input.")
        trainDataMgr.setDataSize(64, 1, 281, 281, 4, "TrainData")  # batchSize, depth, height, width, k
        testDataMgr.setDataSize(64, 1, 281, 281, 4, "TestData")  # batchSize, depth, height, width, k
        net = SegV2DModel()

    else:
        print("Info: program uses 3D input.")
        trainDataMgr.setDataSize(64, 21, 281, 281, 4, "TrainData")  # batchSize, depth, height, width, k
        testDataMgr.setDataSize(64, 21, 281, 281, 4, "TestData")  # batchSize, depth, height, width, k
        net = SegV3DModel()

    trainDataMgr.setMaxShift(25)                  #translation data augmentation
    trainDataMgr.setFlipProb(0.3)                 #flip data augmentation

    net.printParametersScale()
    net.setDropoutProb(0.2)

    ceWeight = torch.FloatTensor([1, 39, 68, 30653])
    lossFunc = FocalCELoss(weight=ceWeight)
    # lossFunc = nn.CrossEntropyLoss(weight=ceWeight)
    # lossFunc = nn.CrossEntropyLoss()
    net.setLossFunc(lossFunc)

    optimizer = optim.Adam(net.parameters())
    net.setOptimizer(optimizer)

    netMgr = NetMgr(net, netPath)
    bestTestDiceList = [0,0,0,0]
    if 2 == len(trainDataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet(True)  # True for train
        bestTestDiceList = netMgr.loadBestTestDice()
        print('Current best test dice: ', bestTestDiceList)
    else:
        print("Network trains from scratch.")

    # print model
    print("\n====================Net Architecture===========================")
    summary(net.cuda(), trainDataMgr.getInputSize())
    print("===================End of Net Architecture =====================\n")

    #===========debug==================
    #trainDataMgr.setOneSampleTraining(True) # for debug
    #testDataMgr.setOneSampleTraining(True)  # for debug
    useDataParallel = True  # for debug
    # ===========debug==================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU >1:
            print(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net)
    net.to(device)

    epochs = 15000
    K = testDataMgr.getNumClassification()
    print("Hints: Test Dice_0 is the dice coeff for all non-zero labels")
    print("Hints: Test Dice_1 is for primary cancer(green), test Dice_2 is for metastasis(yellow), and test Dice_3 is for invaded lymph node(brown).")
    print("Hints: Test TPR_0 is the TPR for all non-zero labels")
    print("Hints: Test TPR_1 is for primary cancer(green), TPR_2 is for metastasis(yellow), and TPR_3 is for invaded lymph node(brown).\n")
    diceHead = (f'Dice_{i}' for i in range(K))
    TPRHead = (f'TPR_{i}' for i in range(K))
    print(f"Epoch \t TrainingLoss \t TestLoss \t", '\t'.join(diceHead),'\t', '\t'.join(TPRHead))   # print output head

    for epoch in range(epochs):

        #================Training===============
        random.seed()
        trainingLoss = 0.0
        batches = 0
        net.train()
        for inputs, labels in trainDataMgr.dataLabelGenerator(True):
            inputs, labels= torch.from_numpy(inputs), torch.from_numpy(labels)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)  # return a copy

            if useDataParallel:
                optimizer.zero_grad()
                outputs = net.forward(inputs)
                loss = lossFunc(outputs, labels)
                loss.backward()
                optimizer.step()
                batchLoss = loss.item()
            else:
                batchLoss = net.batchTrain(inputs, labels)

            trainingLoss += batchLoss
            batches += 1
            #print(f'batch={batches}: batchLoss = {batchLoss}')

        trainingLoss /= batches

        # ================Test===============
        net.eval()
        with torch.no_grad():
            diceSumList = [0 for _ in range(K)]
            diceCountList = [0 for _ in range(K)]
            TPRSumList = [0 for _ in range(K)]
            TPRCountList = [0 for _ in range(K)]
            testLoss = 0.0
            batches = 0
            for inputs, labelsCpu in testDataMgr.dataLabelGenerator(False):
                inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labelsCpu)
                inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)  # return a copy

                if useDataParallel:
                    outputs = net.forward(inputs)
                    loss = lossFunc(outputs, labels)
                    batchLoss = loss.item()
                else:
                    batchLoss, outputs = net.batchTest(inputs, labels)

                outputs = outputs.cpu().numpy()
                segmentations = testDataMgr.oneHotArray2Segmentation(outputs)
                
                (diceSumBatch, diceCountBatch) = testDataMgr.getDiceSumList(segmentations, labelsCpu)
                (TPRSumBatch, TPRCountBatch) = testDataMgr.getTPRSumList(segmentations, labelsCpu)
                
                diceSumList = [x+y for x,y in zip(diceSumList, diceSumBatch)]
                diceCountList = [x+y for x,y in zip(diceCountList, diceCountBatch)]
                TPRSumList = [x + y for x, y in zip(TPRSumList, TPRSumBatch)]
                TPRCountList = [x + y for x, y in zip(TPRCountList, TPRCountBatch)]
                
                testLoss += batchLoss
                batches += 1
                #print(f'batch={batches}: batchLoss = {batchLoss}')

        #===========print train and test progress===============
        testLoss /= batches
        diceAvgList = [x/(y+1e-8) for x,y in zip(diceSumList, diceCountList)]
        TPRAvgList = [x / (y + 1e-8) for x, y in zip(TPRSumList, TPRCountList)]
        print(f'{epoch} \t {trainingLoss:.4f} \t {testLoss:.4f} \t', '\t'.join( (f'{x:.3f}' for x in diceAvgList)),'\t', '\t'.join( (f'{x:.3f}' for x in TPRAvgList)))

        # =============save net parameters==============
        if trainingLoss != float('inf') and trainingLoss != float('nan'):
            netMgr.save(diceAvgList)
            if diceAvgList[1] > 0.50  and diceAvgList[1] > bestTestDiceList[1]:  # compare the primary dice.
                bestTestDiceList = diceAvgList
                netMgr.saveBest(bestTestDiceList)
        else:
            print("Error: training loss is infinity. Program exit.")
            sys.exit()

    torch.cuda.empty_cache()
    print("=============END of Training of Ovarian Cancer Segmentation V Model =================")

if __name__ == "__main__":
    main()
