# support various input size

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import os
import random

import torch
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../..")

from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader
from framework.SurfaceSegNet_Q import SurfaceSegNet_Q
from OCTData.OCTDataSet import  OCTDataSet
from OCTData.OCTDataSet6Bscans import  OCTDataSet6Bscans
from OCTData.OCTTransform import OCTDataTransform
from OCTData.OCTDataUtilities import computeMASDError

def printUsage(argv):
    print("============ Cross Validation Train OCT MultiSurface Network =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def convertTensor2Dict(aTensor):
    result ={}
    N = aTensor.numel()
    for i in range(N):
        result[str(i)] = aTensor[i]
    return result

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")

    if hps.dataIn1Parcel:
        if -1==hps.k and 0==hps.K:  # do not use cross validation
            trainImagesPath = os.path.join(hps.dataDir, "training", f"images.npy")
            trainLabelsPath = os.path.join(hps.dataDir, "training", f"surfaces.npy")
            trainIDPath = os.path.join(hps.dataDir, "training", f"patientID.json")

            if hps.useIndependentValidation:
                validationImagesPath = os.path.join(hps.dataDir, "validation", f"images.npy")
                validationLabelsPath = os.path.join(hps.dataDir, "validation", f"surfaces.npy")
                validationIDPath = os.path.join(hps.dataDir, "validation", f"patientID.json")
            else:
                validationImagesPath = os.path.join(hps.dataDir, "test", f"images.npy")
                validationLabelsPath = os.path.join(hps.dataDir, "test", f"surfaces.npy")
                validationIDPath = os.path.join(hps.dataDir, "test", f"patientID.json")

        else:  # use cross validation
            trainImagesPath = os.path.join(hps.dataDir,"training", f"images_CV{hps.k:d}.npy")
            trainLabelsPath  = os.path.join(hps.dataDir,"training", f"surfaces_CV{hps.k:d}.npy")
            trainIDPath     = os.path.join(hps.dataDir,"training", f"patientID_CV{hps.k:d}.json")

            validationImagesPath = os.path.join(hps.dataDir,"validation", f"images_CV{hps.k:d}.npy")
            validationLabelsPath = os.path.join(hps.dataDir,"validation", f"surfaces_CV{hps.k:d}.npy")
            validationIDPath    = os.path.join(hps.dataDir,"validation", f"patientID_CV{hps.k:d}.json")
    else:
        if -1==hps.k and 0==hps.K:  # do not use cross validation
            trainImagesPath = os.path.join(hps.dataDir, "training", f"patientList.txt")
            trainLabelsPath = None
            trainIDPath = None

            validationImagesPath = os.path.join(hps.dataDir, "validation", f"patientList.txt")
            validationLabelsPath = None
            validationIDPath = None
        else:
            print(f"Current do not support Cross Validation and not dataIn1Parcel\n")
            assert(False)

    trainTransform = OCTDataTransform(prob=hps.augmentProb, noiseStd=hps.gaussianNoiseStd, saltPepperRate=hps.saltPepperRate, saltRate=hps.saltRate, rotation=hps.rotation,  flippingProb=hps.flippingProb)
    validationTransform = trainTransform
    # validation supporting data augmentation benefits both learning rate decaying and generalization.

    trainData = OCTDataSet(trainImagesPath, trainIDPath, trainLabelsPath,  transform=trainTransform, hps=hps)
    validationData = OCTDataSet(validationImagesPath, validationIDPath, validationLabelsPath,  transform=validationTransform,  hps=hps)

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    if hps.optim == "AdamPlateau":
        optimizer = optim.Adam(net.parameters(), lr=hps.learningRate, weight_decay=0)
        lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=hps.lrPatience, min_lr=1e-8, threshold=0.02, threshold_mode='rel')
    if hps.optim == "SGDOneCycle":
        optimizer = torch.optim.SGD(net.parameters(), lr=hps.learningRate, momentum=hps.momentum)
        stepsPerEpoch = trainData.__len__() // hps.batchSize
        lrScheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=hps.maxLr, steps_per_epoch=stepsPerEpoch, epochs=hps.epochs)

    net.setOptimizer(optimizer)
    net.setLrScheduler(lrScheduler)

    # Load network

    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet("train")


    writer = SummaryWriter(log_dir=hps.logDir)

    # train
    epochs = hps.epochs
    preTrainingLoss = 999999.0
    preValidLoss = net.getRunParameter("validationLoss") if "validationLoss" in net.m_runParametersDict else 2041  # float 16 has maxvalue: 2048
    preErrorMean = net.getRunParameter("errorMean") if "errorMean" in net.m_runParametersDict else 10.0
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
        for batchData in data.DataLoader(trainData, batch_size=hps.batchSize, shuffle=True, num_workers=0, drop_last=True):
            trBatch += 1
            S, sigma2, loss, _x = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(hps.device))
            optimizer.step()
            trLoss += float(loss)

            if hps.optim == "SGDOneCycle":
                lrScheduler.step()  # for OneCycleLR

            # break

        trLoss = trLoss / trBatch
        #lrScheduler.step(trLoss)
        # print(f"epoch:{epoch}; trLoss ={trLoss}\n")

        net.eval()
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0
            net.setStatus("validation")
            for batchData in data.DataLoader(validationData, batch_size=hps.batchSize, shuffle=False,
                                                              num_workers=0):
                validBatch += 1
                # S is surface location in (B,S,W) dimension, the predicted Mu
                S, sigma2, loss, _x  = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
                validLoss += float(loss)
                validOutputs = torch.cat((validOutputs, S)) if validBatch != 1 else S
                validGts = torch.cat((validGts, batchData['GTs'])) if validBatch != 1 else batchData['GTs'] # Not Gaussian GTs
                validIDs = validIDs + batchData['IDs'] if validBatch != 1 else batchData['IDs']  # for future output predict images
                # break

            validLoss = validLoss / validBatch
            if hps.groundTruthInteger:
                validOutputs = (validOutputs+0.5).int() # as ground truth are integer, make the output also integers.
            # print(f"epoch:{epoch}; validLoss ={validLoss}\n")

            volumeIDs = []
            volumeBscanStartIndexList = []
            B = len(validIDs)
            for i in range(0, B):  # we need consider the different Bscan numbers for different volumes.
                id = validIDs[i]
                if '_s000' == id[-5:]:
                    volumeIDs.append(id[: id.rfind("_s000")])
                    volumeBscanStartIndexList.append(i)

            stdSurfaceError, muSurfaceError, stdError, muError = computeMASDError(validOutputs, validGts,
                                                                                      volumeBscanStartIndexList,
                                                                                      hPixelSize=hps.hPixelSize)


        if hps.optim == "AdamPlateau":
            lrScheduler.step(validLoss)
        # debug
        # print(f"epoch {epoch} ends...")  # for smoke debug

        writer.add_scalars('Loss', {"train":trLoss, "validation": validLoss}, epoch)
        writer.add_scalar('ValidationError/mean', muError, epoch)
        writer.add_scalar('ValidationError/std', stdError, epoch)
        writer.add_scalars('ValidationError/muSurface', convertTensor2Dict(muSurfaceError), epoch)
        writer.add_scalars('ValidationError/stdSurface', convertTensor2Dict(stdSurfaceError), epoch)
        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        if muError < preErrorMean:
            net.updateRunParameter("validationLoss", validLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            net.updateRunParameter("errorMean", muError)
            net.updateRunParameter("learningRate", optimizer.param_groups[0]['lr'])
            preValidLoss = validLoss
            preErrorMean = muError
            netMgr.saveNet(hps.netPath)

        #break


    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()
