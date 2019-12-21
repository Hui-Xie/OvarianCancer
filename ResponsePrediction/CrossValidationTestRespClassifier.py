#  test cross validation
import sys

import torch
import torch.nn as nn
import yaml
import json
from torch.utils import data

sys.path.append("..")
from FilesUtilities import *
from NetMgr import NetMgr

sys.path.append(".")
from OCDataSet import *
from VoteClassifier import *
from VoteBCEWithLogitsLoss import VoteBCEWithLogitsLoss
from AccuracyFunc import *


def printUsage(argv):
    print("============ Cross Validation Vote Classifier =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")


def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # parse config file
    configFile = sys.argv[1]
    with open(configFile) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    experimentName = getStemName(configFile, removedSuffix=".yaml")
    batchSize = cfg["batchSize"]
    latentDir = cfg["latentDir"]
    suffix = cfg["suffix"]
    patientResponsePath = cfg["patientResponsePath"]
    K_folds = cfg["K_folds"]
    fold_k = cfg["fold_k"]
    netPath = cfg["netPath"] + "/" + experimentName
    rawF = cfg["rawF"]
    F = cfg["F"]
    device = eval(cfg["device"])  # convert string to class object.
    featureIndices = cfg["featureIndices"]

    # prepare data
    dataPartitions = OVDataPartition(latentDir, patientResponsePath, suffix, K_folds=K_folds, k=fold_k)
    positiveWeight = dataPartitions.getPositiveWeight().to(device)
    testData = OVDataSet('test', dataPartitions, preLoadData=True)

    # construct network
    net = VoteClassifier()
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    loss0 = VoteBCEWithLogitsLoss(pos_weight=positiveWeight, weightedVote=False)
    net.appendLossFunc(loss0, 0.5)
    loss1 = nn.BCEWithLogitsLoss(pos_weight=positiveWeight)
    net.appendLossFunc(loss1, 0.5)

    # Load network
    if os.path.exists(netPath) and 2 == len(getFilesList(netPath, ".pt")):
        netMgr = NetMgr(net, netPath, device)
        netMgr.loadNet("test")
        print(f"Fully Conneted Classifier load from  {netPath}")
    else:
        print(f"Error: program can not load network")

    logDir = latentDir + "/log/" + experimentName
    if not os.path.exists(logDir):
        os.makedirs(logDir)  # recursive dir creation
    testResultFilename = os.path.join(logDir, f"testResult_CV{fold_k}.json")
   
    net.eval()
    with torch.no_grad():
        testBatch = 0  # valid means validation
        testLoss = 0.0
        for inputs, labels, patientIDs in data.DataLoader(testData, batch_size=batchSize, shuffle=False,num_workers=0):
            testBatch += 1
            inputs = inputs.to(device, dtype=torch.float)
            gts = labels.to(device, dtype=torch.float)
            outputs, loss = net.forward(inputs, gts=gts)
            testLoss += loss
            testOutputs = torch.cat((testOutputs, outputs)) if testBatch != 1 else outputs
            testGts = torch.cat((testGts, gts)) if testBatch != 1 else gts
            testPatientIDs = testPatientIDs + patientIDs if testBatch != 1 else patientIDs

        testLoss = testLoss / testBatch
        testAccuracy = computeAccuracy(testOutputs, testGts)
        testTPR = computeTPR(testOutputs, testGts)
        testTNR = computeTNR(testOutputs, testGts)

    # print result:
    N = testOutputs.shape[0]
    print(f"In the fold_{fold_k} CV: testSetSize={N}, Accuracy={testAccuracy}, TPR={testTPR}, TNR={testTNR}")
    testResult = {}
    for i in range(N):
        testResult[testPatientIDs[i]] = (1 if testOutputs[i].item() >=0 else 0)
    with open(testResultFilename, "w") as fp:
        json.dump(testResult,fp)
    print(f"result file is in {testResultFilename}")

    print("================ End of Cross Validation Test ==============")


if __name__ == "__main__":
    main()
