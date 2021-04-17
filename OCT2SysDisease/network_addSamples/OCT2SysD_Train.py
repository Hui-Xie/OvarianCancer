# OCT to Systemic Disease training program

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import random
import datetime

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
from OCT2SysD_DataSet import OCT2SysD_DataSet
from OCT2SysD_Transform import OCT2SysD_Transform
from ThicknessMap2HyperTensionNet_C import ThicknessMap2HyperTensionNet_C
from ThicknessMap2HyperTensionNet_D import ThicknessMap2HyperTensionNet_D
from ThicknessMap2Gender_A import ThicknessMap2Gender_A
from ThicknessMap2Gender_B import ThicknessMap2Gender_B
from ThicknessMap2Gender_ResNet import ThicknessMap2Gender_ResNet
from ThicknessMap2HyperTensionNet_VGG16 import ThicknessMap2HyperTensionNet_VGG16
from ThicknessMap2HyperTensionNet_HalfUNet import ThicknessMap2HyperTensionNet_HalfUNet
from ThicknessMap2HyperTensionNet_VGG16BatchNorm import ThicknessMap2HyperTensionNet_VGG16BatchNorm
from ThicknessMap2HyperTensionNet_VGG16C import ThicknessMap2HyperTensionNet_VGG16C
from ThicknessMap2HyperTensionNet_E import ThicknessMap2HyperTensionNet_E

# for input size: 9x31x25
from ThicknessMap2HyTension_ResNet import ThicknessMap2HyTension_ResNet
from Thickness2HyTensionNet_1Layer import Thickness2HyTensionNet_1Layer

from OCT2SysD_Tools import *

sys.path.append("../..")
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.measure import  *


def printUsage(argv):
    print("============ Training of thickness map to binary prediction Network =============")
    print("=======input data is thickness enface map ===========================")
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

    trainTransform = OCT2SysD_Transform(hps) if hps.trainAugmentation else None
    validationTransform = OCT2SysD_Transform(hps) if hps.validationAugmentation else None
    # some people think validation supporting data augmentation benefits both learning rate decaying and generalization.

    trainData = OCT2SysD_DataSet("training", hps=hps, transform=trainTransform)
    validationData = OCT2SysD_DataSet("validation", hps=hps, transform=validationTransform)

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # optimizer = optim.Adam(net.parameters(), lr=hps.learningRate, weight_decay=0)

    # refer to mobileNet v3 paper, use RMSprop optimizer
    # optimizer = optim.RMSprop(net.parameters(), lr=hps.learningRate, weight_decay=0, momentum=0.9)

    # adaptive optimizers sometime are worse than SGD
    '''
    https://arxiv.org/abs/1705.08292
    The Marginal Value of Adaptive Gradient Methods in Machine Learning
    Adaptive optimization methods, which perform local optimization with a metric constructed from the history of iterates, 
    are becoming increasingly popular for training deep neural networks. Examples include AdaGrad, RMSProp, and Adam. 
    We show that for simple overparameterized problems, adaptive methods often find drastically different solutions 
    than gradient descent (GD) or stochastic gradient descent (SGD). We construct an illustrative binary classification 
    problem where the data is linearly separable, GD and SGD achieve zero test error, and AdaGrad, Adam, and RMSProp attain 
    test errors arbitrarily close to half. We additionally study the empirical generalization capability of adaptive methods
    on several state-of-the-art deep learning models. We observe that the solutions found by adaptive methods generalize 
    worse (often significantly worse) than SGD, even when these solutions have better training performance. 
    These results suggest that practitioners should reconsider the use of adaptive methods to train neural networks.
    '''
    optimizer = optim.SGD(net.parameters(), lr=hps.learningRate, weight_decay=hps.weightDecay, momentum=0)
    net.setOptimizer(optimizer)

    # lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-8, threshold=0.02, threshold_mode='rel')

    # math.log(0.5,0.98) = 34, this scheduler equals scale 0.5 per 100 epochs.
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=hps.lrSchedulerMode, factor=hps.lrDecayFactor, patience=hps.lrPatience, min_lr=1e-5, threshold=0.015, threshold_mode='rel')
    net.setLrScheduler(lrScheduler)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("train")

    writer = SummaryWriter(log_dir=hps.logDir)

    # train
    epochs = 1360000
    preValidLoss = net.getRunParameter("validationLoss") if "validationLoss" in net.m_runParametersDict else 1e+8  # float 16 has maxvalue: 2048
    preAccuracy = 0.5
    if net.training:
        initialEpoch = net.getRunParameter("epoch") if "epoch" in net.m_runParametersDict else 0
    else:
        initialEpoch = 0

    for epoch in range(initialEpoch, epochs):
        random.seed()
        net.m_epoch = epoch
        net.setStatus("training")

        net.train()
        trBatch = 0
        trLoss = 0.0

        allTrainOutput = None
        allTrainGTs = None

        for batchData in data.DataLoader(trainData, batch_size=hps.batchSize, shuffle=True, num_workers=0, drop_last=True):
            inputs = batchData['images']# B,C,H,W
            t = batchData['GTs'].to(device=hps.device, dtype=torch.float) # target

            x, loss = net.forward(inputs, t)
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(hps.device))
            optimizer.step()

            trLoss += float(loss)
            trBatch += 1

            allTrainOutput = x if allTrainOutput is None else torch.cat((allTrainOutput, x))
            allTrainGTs = t if allTrainGTs is None else torch.cat((allTrainGTs, t))

            #debug
            # break

        trAcc = computeClassificationAccuracyWithLogit(allTrainGTs, allTrainOutput)

        trLoss /= trBatch

        net.eval()
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0

            allValidationOutput = None
            allValidationGTs = None

            net.setStatus("validation")
            for batchData in data.DataLoader(validationData, batch_size=hps.batchSize, shuffle=False, num_workers=0):
                inputs = batchData['images']  # B,C,H,W
                t = batchData['GTs'].to(device=hps.device, dtype=torch.float)  # target

                x, loss = net.forward(inputs, t)
                validLoss += float(loss)
                validBatch += 1

                allValidationOutput = x if allValidationOutput is None else torch.cat((allValidationOutput, x))
                allValidationGTs    = t if allValidationGTs    is None else torch.cat((allValidationGTs,    t))

                # debug
                # break

            validLoss /= validBatch

        validAcc = computeClassificationAccuracyWithLogit(allValidationGTs, allValidationOutput)
        Td_Acc_TPR_TNR_Sum = search_Threshold_Acc_TPR_TNR_Sum_WithLogits(allValidationGTs, allValidationOutput)


        if "min" == hps.lrSchedulerMode:
            lrScheduler.step(validLoss)
        else: # "max"
            lrScheduler.step(Td_Acc_TPR_TNR_Sum['Sum'])

        writer.add_scalars('loss',     {"train": trLoss, "validation": validLoss}, epoch)
        writer.add_scalars('accuracy', {"train": trAcc, "validation": validAcc}, epoch)

        writer.add_scalar('train/Loss', trLoss, epoch)
        writer.add_scalar('train/Accuracy', trAcc, epoch)
        writer.add_scalar('validation/Loss', validLoss, epoch)
        writer.add_scalar('validation/Accuracy', validAcc, epoch)
        writer.add_scalars('validation/threshold_ACC_TPR_TNR_Sum', Td_Acc_TPR_TNR_Sum, epoch)
        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        #if validLoss < preValidLoss:
        # if  validAcc > preAccuracy:
        if Td_Acc_TPR_TNR_Sum['Sum'] > preAccuracy:
            net.updateRunParameter("validationLoss", validLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            net.updateRunParameter("accuracy", validAcc)
            net.updateRunParameter("learningRate", optimizer.param_groups[0]['lr'])
            preValidLoss = validLoss
            preAccuracy = Td_Acc_TPR_TNR_Sum['Sum']
            netMgr.saveNet(hps.netPath)


        # debug
        # print(f"smoke test: finish one epoch of training and  validation")

    print("============ End of Training Thickness enface map 2 Binary Systemic Disease Prediction ===========")



if __name__ == "__main__":
    main()
