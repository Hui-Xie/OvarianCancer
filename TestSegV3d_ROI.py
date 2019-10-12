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
      1  test SegV3d model for all validation and test files;
      2  output the dice coefficient for each file;

Discarded changes:                  

Experiment setting:
Input CT data: 51*171*171 ROI around primary cancer

Loss Function:  SoftMax

Data:   total 36 patients with 50-80% label, 6-fold cross validation, test 6, validation 6, and training 24.  
    script: python3.7 statisticsLabelFiles.py 
    Total 36 in /home/hxie1/data/OvarianCancerCT/primaryROI/labels_npy
    0 has 48159408 elements, with a rate of  0.8970491562903105 
    1 has 5527068 elements, with a rate of  0.10295084370968957

Training strategy: 

          '''


def printUsage(argv):
    print("============Test Seg 3D VNet for ROI around primary Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <predictOutputDir> <fullPathOfData>  <fullPathOfLabel> <k>  <GPUID_List>")
    print("where: \n"
          "       netSavedPath must be specific network directory.\n"
          "       k=[0, K), the k-th fold in the K-fold cross validation.\n"
          "       GPUIDList: 0,1,2,3, the specific GPU ID List, separated by comma\n")


def main():
    if len(sys.argv) != 7:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    predictOutputDir = sys.argv[2]
    dataInputsPath = sys.argv[3]
    groundTruthPath = sys.argv[4]
    k = int(sys.argv[5])
    GPUIDList = sys.argv[6].split(',')  # choices: 0,1,2,3 for lab server.
    GPUIDList = [int(x) for x in GPUIDList]
    useLabelConsistencyLoss = False
    # ===========debug==================
    useDataParallel = True if len(GPUIDList) > 1 else False  # for debug
    # ===========debug==================

    print(f'Program ID:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'.........')

    inputSuffix = ".npy"
    K_fold = 6
    batchSize = 2 * len(GPUIDList)-1
    print(f"batchSize = {batchSize}")
    numWorkers = 0

    device = torch.device(f"cuda:{GPUIDList[0]}" if torch.cuda.is_available() else "cpu")

    timeStr = getStemName(netPath)
    if timeStr == "Best":
        timeStr = getStemName(netPath.replace("/Best", ""))
    logFile = os.path.join(predictOutputDir, f"predict_CV{k:d}_{timeStr}.txt")
    print(f'Test log is in {logFile}')

    logging.basicConfig(filename=logFile, filemode='a+', level=logging.INFO, format='%(message)s')

    dataPartitions = OVDataSegPartition(dataInputsPath, groundTruthPath, inputSuffix, K_fold, k)
    validationTransform = OCDataLabelTransform(0)
    testTransform = OCDataLabelTransform(0)

    validationData = OVDataSegSet('validation', dataPartitions, transform=validationTransform)
    testData = OVDataSegSet('test', dataPartitions, transform=testTransform)

    net = SegV3DModel(useLabelConsistencyLoss=useLabelConsistencyLoss)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    # Load network
    netMgr = NetMgr(net, netPath, device)

    if 2 == len(getFilesList(netPath, ".pt")):
        netMgr.loadNet("test")
        bestTestPerf = netMgr.loadBestTestPerf()
        print(f"Best validation dice: {bestTestPerf}")
    else:
        print (f"Error net path, program can not load")
        return -2

    if useDataParallel:
        net = nn.DataParallel(net, device_ids=GPUIDList, output_device=device)

    logging.info(f"ID" + f"\t\tDice")  # logging.info output head


    # ================Validation===============
    net.eval()

    with torch.no_grad():
        for inputs, labels, patientIDs in data.DataLoader(validationData, batch_size=batchSize, shuffle=False,
                                                          num_workers=numWorkers):
            inputs = inputs.to(device, dtype=torch.float)
            gts = labels.to(device, dtype=torch.float)  # return a copy
            gts = (gts > 0).long()  # not discriminate all non-zero labels.

            outputs, _ = net.forward(inputs, gts)

            # compute dice
            gtsShape = gts.shape
            for i in range(gtsShape[0]):
                output = torch.argmax(outputs[i,], dim=0)
                gt = gts[i,]
                dice = tensorDice(output, gt)
                filename = os.path.join(predictOutputDir, patientIDs[i]+".npy")
                np.save(filename, output.cpu().numpy())
                logging.info(patientIDs[i] +f"\t{dice:.5f}")

    # ================Independent Test===============
    net.eval()
    with torch.no_grad():
        for inputs, labels, patientIDs in data.DataLoader(testData, batch_size=batchSize, shuffle=False,
                                                          num_workers=numWorkers):
            inputs = inputs.to(device, dtype=torch.float)
            gts = labels.to(device, dtype=torch.float)  # return a copy
            gts = (gts > 0).long()  # not discriminate all non-zero labels.

            outputs, _ = net.forward(inputs, gts)

            # compute dice
            gtsShape = gts.shape
            for i in range(gtsShape[0]):
                output = torch.argmax(outputs[i,], dim=0)
                gt = gts[i,]
                dice = tensorDice(output, gt)
                filename = os.path.join(predictOutputDir, patientIDs[i] + ".npy")
                np.save(filename, output.cpu().numpy())
                logging.info(patientIDs[i] + f"\t{dice:.5f}")

    torch.cuda.empty_cache()
    print(f"\n\n=============END of Test of SegV3d ROI Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    print(f'\nProgram Ending Time: {str(curTime)}')


if __name__ == "__main__":
    main()
