# Ovarian cancer training program

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import datetime

import torch

sys.path.append(".")
from OVDataSet_TTA import OVDataSet_TTA
from ResponseNet import ResponseNet
from OVTools import *

sys.path.append("../..")
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader


def printUsage(argv):
    print("============ TestTimeAugmentation Test Ovarian Cancer Response Network =============")
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
    print(f"TTA Test Experiment: {hps.experimentName}")

    testData = OVDataSet_TTA("test", hps=hps, transform=None) # only use data augmentation at training set

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("test")

    net.m_epoch = -1

    net.eval()
    predictProbDict={}
    with torch.no_grad():
        testBatch = 0
        testLoss = 0.0
        testResidualLoss = 0.0
        testChemoLoss = 0.0
        testAgeLoss = 0.0
        testSurvivalLoss = 0.0
        testOptimalLoss = 0.0

        net.setStatus("test")
        for batchData in testData.generator():
            residualPredict, residualLoss, chemoPredict, chemoLoss, agePredict, ageLoss, survivalPredict, survivalLoss, optimalPredict, optimalLoss, predictProb\
                = net.forward(batchData['images'], GTs=batchData['GTs'])

            loss = hps.lossWeights[0]*residualLoss + hps.lossWeights[1]*chemoLoss + hps.lossWeights[2]*ageLoss \
                    + hps.lossWeights[3]*survivalLoss  + hps.lossWeights[4]*optimalLoss

            testLoss += loss
            testResidualLoss += residualLoss
            testChemoLoss += chemoLoss
            testAgeLoss += ageLoss
            testSurvivalLoss += survivalLoss
            testOptimalLoss += optimalLoss
            testBatch +=1

            MRN = batchData['IDs'][0]  # [0] is for list to string
            B = len(batchData['IDs'])
            predictProbDict[MRN] = {}
            predictProbDict[MRN]['Prob1'] = predictProb.sum()/B
            predictProbDict[MRN]['GT'] = batchData['GTs']['OptimalResult'][0]
            print(f"MRN={MRN}; predictProb = {predictProb.view(B)}")

    curTime = datetime.datetime.now()
    timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
    outputPredictProbDict2Csv(predictProbDict, hps.outputDir + f"/testSetPredictProb_{timeStr}.csv")

    print("============ End of TTA test of Ovarian Cancer  Network ===========")



if __name__ == "__main__":
    main()
