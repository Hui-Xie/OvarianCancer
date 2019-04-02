import sys
from DataMgr import DataMgr
from SegVModel import SegVModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def printUsage(argv):
    print("============Train Ovarian Cancer Segmentation V model=============")
    print("Usage:")
    print(argv[0], " <fullPathOfTrainImages>  <fullPathOfTrainLabels>")

def main():
    if len(sys.argv) != 3:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    dataMgr = DataMgr(sys.argv[1], sys.argv[2])
    dataMgr.setDataSize(4, 21,281,281,4)  #batchSize, depth, height, width, k

    net= SegVModel()
    net.printParamtersScale()

    lossFunc = nn.CrossEntropyLoss()
    net.setLossFunc(lossFunc)

    optimizer = optim.Adam(net.parameters())
    net.setOptimizer(optimizer)

    dataMgr.setOneSampleTraining(True) # for debug
    useDataParallel = True  # for debug

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU >1:
            print(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net)
    net.to(device)

    epochs = 2
    for epoch in range(epochs):
        runningLoss = 0.0
        batches = 0
        for inputs, labels in dataMgr.dataLabelGenerator(True):
            inputs, labels= torch.from_numpy(inputs), torch.from_numpy(labels)
            inputs, labels = inputs.to(device, dtype=torch.float), labels.to(device, dtype=torch.long)  # return a copy

            batchLoss = 0.0
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
            print(f'batch={batches}: batchLoss = {batchLoss}')

        epochLoss = runningLoss/batches
        print(f'Epoch={epoch}: epochLoss={epochLoss}')

    print("=============END Training of Ovarian Cancer Segmentation V model =================")

if __name__ == "__main__":
    main()
