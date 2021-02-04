# OCT to Systemic Disease Test program

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import datetime
import os

import torch
from torch.utils import data

sys.path.append(".")
from OCT2SysD_DataSet import OCT2SysD_DataSet
from OCT2SysD_Tools import *
from ThicknessClinical29Ftrs_FCNet import ThicknessClinical29Ftrs_FCNet
from ThicknessClinical29Ftrs_FCNet_B import ThicknessClinical29Ftrs_FCNet_B
from ThicknessClinical29Ftrs_FCNet_C import ThicknessClinical29Ftrs_FCNet_C

sys.path.append("../..")
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.measure import  *


def printUsage(argv):
    print("============ Test of OCT to Systemic Disease Network =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")


def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")

    testData = OCT2SysD_DataSet("test", hps=hps, transform=None) # only use data augmentation for training set

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("test")
 

    net.eval()
    # predictProbDict={}

    threshold = net.getRunParameter("threshold") if "threshold" in net.m_runParametersDict else hps.threshold

    with torch.no_grad():
        testBatch = 0  # valid means validation
        testLoss = 0.0

        allTestOutput = None
        allTestGTs = None

        net.setStatus("test")
        for batchData in data.DataLoader(testData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
            inputs = batchData['images']  # B,C,H,W
            t = batchData['GTs'].to(device=hps.device, dtype=torch.float)  # target

            x, loss = net.forward(inputs, t)
            testLoss += loss
            testBatch += 1

            allTestOutput = x if allTestOutput is None else torch.cat((allTestOutput, x))
            allTestGTs = t if allTestGTs is None else torch.cat((allTestGTs, t))

            # debug
            # break

        testLoss /= testBatch


    Acc_TPR_TNR_Sum = compute_Acc_TPR_TNR_Sum_WithLogits(allTestGTs, allTestOutput,threshold)


    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
    # outputPredictProbDict2Csv(predictProbDict, hps.outputDir + f"/testSetPredictProb_{timeStr}.csv")

    with open(os.path.join(hps.outputDir, f"testOutput_{timeStr}.txt"), "w") as file:
        hps.printTo(file)
        file.write("\n=======net running parameters=========\n")
        file.write(f"net.m_runParametersDict:\n")
        [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]
        file.write(f"\n=======Test Result on test data with threshold of {threshold}=========\n")
        [file.write(f"\t{key}:{value}\n") for key, value in Acc_TPR_TNR_Sum.items()]

        file.write(f"\n=======Test Result on test data with threshold of 0.5 =========\n")
        Acc_TPR_TNR_Sum2 = compute_Acc_TPR_TNR_Sum_WithLogits(allTestGTs, allTestOutput, 0.5)
        [file.write(f"\t{key}:{value}\n") for key, value in Acc_TPR_TNR_Sum2.items()]



    print("============ End of Test OCT2SysDisease Network ===========")



if __name__ == "__main__":
    main()
