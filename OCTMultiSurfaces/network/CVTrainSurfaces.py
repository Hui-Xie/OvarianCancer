# support various input size

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import yaml
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append(".")
from OCTDataSet import *
from OCTUnetTongren import OCTUnetTongren
from OCTUnetJHU import OCTUnetJHU
from OCTUnetSurfaceLayerJHU import OCTUnetSurfaceLayerJHU
from IVUSUnet import IVUSUnet
from SurfacesUnet import SurfacesUnet
from SurfacesUnet_YufanHe import SurfacesUnet_YufanHe
from OCTOptimization import *
from OCTTransform import *

sys.path.append("../..")
from utilities.FilesUtilities import *
from utilities.TensorUtilities import *
from framework.NetMgr import NetMgr
from framework.ConfigReader import ConfigReader


def printUsage(argv):
    print("============ Cross Validation Train OCT MultiSurface Network =============")
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

    if hps.dataIn1Parcel:
        if -1==hps.k and 0==hps.K:  # do not use cross validation
            trainImagesPath = os.path.join(hps.dataDir, "training", f"images.npy")
            trainLabelsPath = os.path.join(hps.dataDir, "training", f"surfaces.npy")
            trainIDPath = os.path.join(hps.dataDir, "training", f"patientID.json")

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

    tainTransform = OCTDataTransform(prob=hps.augmentProb, noiseStd=hps.gaussianNoiseStd, saltPepperRate=hps.saltPepperRate, saltRate=hps.saltRate, rotation=hps.rotation)
    validationTransform = tainTransform
    # validation supporting data augmentation benefits both learning rate decaying and generalization.

    trainData = OCTDataSet(trainImagesPath, trainIDPath, trainLabelsPath,  transform=tainTransform, hps=hps)
    validationData = OCTDataSet(validationImagesPath, validationIDPath, validationLabelsPath,  transform=validationTransform,  hps=hps)

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    optimizer = optim.Adam(net.parameters(), lr=hps.learningRate1, weight_decay=0)
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-8, threshold=0.02, threshold_mode='rel')
    net.setLrScheduler(lrScheduler)

    # Load network
    if os.path.exists(hps.netPath) and len(getFilesList(hps.netPath, ".pt")) >= 2 :
        netMgr = NetMgr(net, hps.netPath, hps.device)
        netMgr.loadNet("train")
        print(f"Network load from  {hps.netPath}")
    else:
        netMgr = NetMgr(net, hps.netPath, hps.device)
        print(f"Net starts training from scratch, and save at {hps.netPath}")

    writer = SummaryWriter(log_dir=hps.logDir)

    # train
    epochs = 1360000
    preTrainingLoss = 999999.0
    preValidLoss = net.getRunParameter("validationLoss") if "validationLoss" in net.m_runParametersDict else 2041  # float 16 has maxvalue: 2048
    preErrorMean = net.getRunParameter("errorMean") if "errorMean" in net.m_runParametersDict else 4.3
    if net.training:
        initialEpoch = net.getRunParameter("epoch") if "epoch" in net.m_runParametersDict else 0
    else:
        initialEpoch = 0

    pre2ndValidLoss = 2041  # save the after pertrain epoch with best loss
    pre2ndErrorMean = 4.3

    if initialEpoch > hps.epochsPretrain:  # for pretrain epochs.
        net.resetLrScheduler(hps.learningRate2)

    for epoch in range(initialEpoch, epochs):
        random.seed()
        net.m_epoch = epoch
        net.setStatus("training")

        if epoch == hps.epochsPretrain:  # for pretrain epochs.
            net.resetLrScheduler(hps.learningRate2)

        net.train()
        trBatch = 0
        trLoss = 0.0
        for batchData in data.DataLoader(trainData, batch_size=hps.batchSize, shuffle=True, num_workers=0):
            trBatch += 1
            forwardOutput = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
            if hps.debug and (hps.useRiftInPretrain or (not net.inPretrain())):
                _S, loss, R = forwardOutput
            else:
                _S, loss = forwardOutput

            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(hps.device))
            optimizer.step()
            trLoss += loss

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
                forwardOutput = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
                if hps.debug and (hps.useRiftInPretrain or (not net.inPretrain())):
                    S, loss, R = forwardOutput
                else:
                    S, loss = forwardOutput

                validLoss += loss
                validOutputs = torch.cat((validOutputs, S)) if validBatch != 1 else S
                validGts = torch.cat((validGts, batchData['GTs'])) if validBatch != 1 else batchData['GTs'] # Not Gaussian GTs
                validIDs = validIDs + batchData['IDs'] if validBatch != 1 else batchData['IDs']  # for future output predict images

            validLoss = validLoss / validBatch
            if hps.groundTruthInteger:
                validOutputs = (validOutputs+0.5).int() # as ground truth are integer, make the output also integers.
            # print(f"epoch:{epoch}; validLoss ={validLoss}\n")

            goodBScansInGtOrder =None
            if "OCT_Tongren" in hps.dataDir and 0 != len(hps.goodBscans):
                # example: "/home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
                goodBScansInGtOrder = []
                b = 0
                while b < len(validIDs):
                    patientPath, filename = os.path.split(validIDs[b])
                    patientIDVolumeName = os.path.basename(patientPath)
                    patientID = int(patientIDVolumeName[0:patientIDVolumeName.find("_OD_")])
                    lowB = hps.goodBscans[patientID][0]-1
                    highB = hps.goodBscans[patientID][1]
                    goodBScansInGtOrder.append([lowB,highB])
                    b += hps.slicesPerPatient #validation data and test data both have 31 Bscans per patient

            stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(validOutputs, validGts,
                                                                                                 slicesPerPatient=hps.slicesPerPatient,
                                                                                                 hPixelSize=hps.hPixelSize,
                                                                                                 goodBScansInGtOrder=goodBScansInGtOrder)

        lrScheduler.step(validLoss)
        # debug
        # print(f"epoch {epoch} ends...")  # for smoke debug

        writer.add_scalar('Loss/train', trLoss, epoch)
        writer.add_scalar('Loss/validation', validLoss, epoch)
        writer.add_scalar('ValidationError/mean', muError, epoch)
        writer.add_scalar('ValidationError/std', stdError, epoch)
        writer.add_scalars('ValidationError/muSurface', convertTensor2Dict(muSurfaceError), epoch)
        writer.add_scalars('ValidationError/stdSurface', convertTensor2Dict(stdSurfaceError), epoch)
        writer.add_scalar('learningRate', optimizer.param_groups[0]['lr'], epoch)

        if validLoss < preValidLoss or muError < preErrorMean:
            net.updateRunParameter("validationLoss", validLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            net.updateRunParameter("errorMean", muError)
            preValidLoss = validLoss
            preErrorMean = muError
            netMgr.saveNet(hps.netPath)

        # save the after pertrain epoch with best loss
        if (epoch >= hps.epochsPretrain) and (validLoss < pre2ndValidLoss or muError < pre2ndErrorMean):
            # save realtime network parameter
            bestRunParametersDict = net.m_runParametersDict.copy()
            net.updateRunParameter("validationLoss", validLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            net.updateRunParameter("errorMean", muError)
            pre2ndValidLoss = validLoss
            pre2ndErrorMean = muError
            netMgr.saveRealTimeNet(hps.netPath)
            net.m_runParametersDict = bestRunParametersDict



    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()
