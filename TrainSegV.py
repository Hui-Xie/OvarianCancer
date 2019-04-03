import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from DataMgr import DataMgr
from SegVModel import SegVModel
from NetMgr  import NetMgr

def printUsage(argv):
    print("============Train Ovarian Cancer Segmentation V model=============")
    print("Usage:")
    print(argv[0], "<netSavedPath> <fullPathOfTrainImages>  <fullPathOfTrainLabels>")

def main():
    if len(sys.argv) != 4:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    trainDataMgr = DataMgr(sys.argv[2], sys.argv[3])
    trainDataMgr.setDataSize(4, 21,281,281,4)  #batchSize, depth, height, width, k

    testImagesDir, testLabelsDir = trainDataMgr.getTestDirs()
    testDataMgr = DataMgr(testImagesDir, testLabelsDir)
    testDataMgr.setDataSize(4, 21, 281, 281, 4)  # batchSize, depth, height, width, k


    net= SegVModel()
    net.printParamtersScale()

    lossFunc = nn.CrossEntropyLoss()
    net.setLossFunc(lossFunc)

    optimizer = optim.Adam(net.parameters())
    net.setOptimizer(optimizer)

    netMgr = NetMgr(net)
    if 0 != len(os.listdir(netPath)):
        netMgr.loadNet(netPath, True)  # True for train
    else:
        print("Network train from scratch.")
    #===========debug==================
    trainDataMgr.setOneSampleTraining(True) # for debug
    testDataMgr.setOneSampleTraining(True)  # for debug
    useDataParallel = True  # for debug
    # ===========debug==================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU >1:
            print(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net)
    net.to(device)

    epochs = 2
    for epoch in range(epochs):

        #================Training===============
        runningLoss = 0.0
        batches = 0
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

            runningLoss += batchLoss
            batches += 1
            #print(f'batch={batches}: batchLoss = {batchLoss}')

        epochLoss = runningLoss/batches
        print(f'Epoch={epoch}: TrainingEpochLoss={epochLoss}')

        # save net parameters
        if epochLoss != float('inf') and epochLoss != float('nan'):
            netMgr.saveNet(netPath)
        else:
            print("Error: training loss is infinity. Program exit.")
            sys.exit()

        # ================Test===============
        runningLoss = 0.0
        batches = 0
        for inputs, labels in testDataMgr.dataLabelGenerator(False):
            inputs, labels = torch.from_numpy(inputs), torch.from_numpy(labels)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)  # return a copy

            if useDataParallel:
                outputs = net.forward(inputs)
                loss = lossFunc(outputs, labels)
                batchLoss = loss.item()
            else:
                batchLoss = net.batchTest(inputs, labels)

            runningLoss += batchLoss
            batches += 1
            #print(f'batch={batches}: batchLoss = {batchLoss}')

        epochLoss = runningLoss / batches
        print(f'Epoch={epoch}: TestEpochLoss={epochLoss}')

    torch.cuda.empty_cache()
    print("=============END Training of Ovarian Cancer Segmentation V model =================")

if __name__ == "__main__":
    main()
