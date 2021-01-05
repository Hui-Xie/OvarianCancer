# OCT to Systemic Disease Test program

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import datetime
import os

import torch
from torch.utils import data

sys.path.append(".")
from OCT2SysD_DataSet import OCT2SysD_DataSet
from OCT2SysD_Net import OCT2SysD_Net
from OCT2SysD_Tools import *

sys.path.append("../..")
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.measure import  *


def printUsage(argv):
    print("============ Test of OCT to Systemic Disease Network =============")
    print("=======input data is random single slice ===========================")
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
 
    # application specific parameter
    hyptertensionPosWeight = torch.tensor(hps.class01Percent[0] / hps.class01Percent[1]).to(hps.device)
    appKey = 'hypertension_bp_plus_history$'

    net.eval()
    predictDict= {}
    predictProbDict={}
    with torch.no_grad():
        testBatch = 0  # test means testation
        testHyperTLoss = 0.0

        net.setStatus("test")
        batchSize = 1 if hps.TTA else hps.batchSize

        for batchData in data.DataLoader(testData, batch_size=batchSize, shuffle=False, num_workers=0):

            # squeeze the extra dimension of data
            inputs = batchData['images'].squeeze(dim=0)
            batchData["GTs"]= {key: batchData["GTs"][key].squeeze(dim=0) for key in batchData["GTs"]}
            batchData['IDs'] = [v[0] for v in batchData['IDs']]  # erase tuple wrapper fot TTA

            x = net.forward(inputs)
            predict, predictProb, loss = net.computeBinaryLoss(x, GTs=batchData['GTs'], GTKey=appKey, posWeight=hyptertensionPosWeight)

            testHyperTLoss +=loss
            testBatch += 1

            B = predictProb.shape[0]
            for i in range(B):
                ID = int(batchData['IDs'][i])  # [0] is for list to string
                predictDict[ID]={}
                predictDict[ID][appKey] = predict[i].item()

                predictProbDict[ID] = {}
                predictProbDict[ID]['Prob1'] = predictProb[i]
                predictProbDict[ID]['GT'] = batchData['GTs'][appKey][i]



        testHyperTLoss /= testBatch

        gtDict = testData.getGTDict()
        hyperTAcc = computeClassificationAccuracy(gtDict,predictDict, appKey)
        Td_Acc_TPR_TNR_Sum = computeThresholdAccTPR_TNRSumFromProbDict(predictProbDict)



        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
        outputPredictProbDict2Csv(predictProbDict, hps.outputDir + f"/testSetPredictProb_{timeStr}.csv")

        with open(os.path.join(hps.outputDir, f"output_{timeStr}.txt"), "w") as file:
            hps.printTo(file)
            file.write("\n=======net running parameters=========\n")
            file.write(f"net.m_runParametersDict:\n")
            [file.write(f"\t{key}:{value}\n") for key, value in net.m_runParametersDict.items()]
            file.write("\n=======Test Result=========\n")
            file.write(f"threshold =0.5, hypertenson accuracy = {hyperTAcc}\n")
            file.write("Best accuracy sum: \n")
            [file.write(f"\t{key}:{value}\n") for key, value in Td_Acc_TPR_TNR_Sum.items()]



    print("============ End of Test OCT2SysDisease Network ===========")



if __name__ == "__main__":
    main()
