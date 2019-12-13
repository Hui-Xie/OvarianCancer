# basing on chosen features, use a fully connected network to classify response
# features indices:
featureIndices = [7, 10, 17, 21, 25, 28, 30, 32, 45, 52, 54, 79, 84, 93, 127, 128, 135, 160, 172, 174, 178, 182, 199,
                   203, 213, 224, 249, 250, 253, 260, 273, 278, 283, 286, 307, 333, 336, 344, 349, 356, 359, 371, 373,
                   375, 380, 382, 386, 411, 415, 426, 432, 436, 441, 448, 450, 451, 456, 459, 462, 465, 469, 479, 482,
                   495, 507, 541, 542, 543, 546, 548, 552, 562, 563, 578, 582, 587, 597, 598, 616, 617, 618, 629, 636,
                   639, 648, 662, 670, 677, 681, 684, 685, 688, 704, 713, 720, 723, 736, 739, 748, 755, 781, 785, 792,
                   834, 838, 840, 865, 870, 874, 875, 876, 879, 891, 901, 902, 903, 914, 922, 923, 947, 948, 955, 957,
                   980, 997, 998, 1018, 1024, 1025, 1026, 1029, 1033, 1044, 1048, 1051, 1066, 1077, 1078, 1092, 1110,
                   1113, 1119, 1137, 1151, 1169, 1172, 1177, 1191, 1198, 1204, 1206, 1207, 1220, 1226, 1231, 1234, 1243,
                   1247, 1257, 1267, 1276, 1297, 1308, 1309, 1338, 1342, 1345, 1357, 1364, 1367, 1368, 1370, 1409, 1417,
                   1418, 1419, 1426, 1429, 1442, 1443, 1454, 1462, 1473, 1480, 1484, 1490, 1499, 1503, 1507, 1518, 1528,
                   1533]
# There are 192 features whose single-feature response prediction accuracy > 0.68, chosen from training data

trainLatentDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/training/latent/latent_20191210_024607"
testLatentDir  = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/latent/latent_20191210_024607"
suffix = '.npy'
patientResponsePath = "/home/hxie1/data/OvarianCancerCT/patientResponseDict.json"
netPath = "/home/hxie1/temp_netParameters/OvarianCancer/FCClassifier"

F=1536  # full feature length of a latent vector
extractF = 192 # lenght of extacted features
device = 1   #GPU ID

import json
import os
from FilesUtilities import *
import numpy as np
import torch
import torch.nn as nn
from FCClassifier import FCClassifier
import torch.optim as optim
from NetMgr import NetMgr
import datetime

# extract feature and ground truth


def loadXY(latentDir, patientResponse):
    originalCwd = os.getcwd()
    os.chdir(latentDir)
    trainingFilesList = [os.path.abspath(x) for x in os.listdir(latentDir) if suffix in x]
    os.chdir(originalCwd)

    N  = len(trainingFilesList)
    X = torch.zeros((N, extractF), dtype=torch.float, device=device, requires_grad=False)
    Y = torch.zeros((N, 1), dtype=torch.int, device=device, requires_grad=False)
    for i, filePath in enumerate(trainingFilesList):
        patientID = getStemName(filePath, suffix)[:8]
        V = np.load(filePath)
        assert (F,) == V.shape
        X[i, :] = torch.from_numpy(V[featureIndices])
        Y[i, 0] = patientResponse[patientID]
    return X, Y

##################### main program ##############################

with open(patientResponsePath) as f:
    patientResponse = json.load(f)

trainingX, trainingY = loadXY(trainLatentDir, patientResponse)
testX,     testY     = loadXY(testLatentDir,  patientResponse)

net = FCClassifier()
# Important:
# If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
# Parameters of a model after .cuda() will be different objects with those before the call.
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0)
net.setOptimizer(optimizer)

loss = nn.BCELoss()
net.appendLossFunc(loss, 1)

# Load network
curTime = datetime.datetime.now()
timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"
netPath = os.path.join(netPath, timeStr)
netMgr = NetMgr(net, netPath, device)

if 2 == len(getFilesList(netPath, ".pt")):
    netMgr.loadNet("train")

epochs = 15000000
oldTestLoss = 100000

print("We are here")


# run network

# print out


