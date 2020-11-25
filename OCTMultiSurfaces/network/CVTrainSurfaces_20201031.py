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
from OCTDataSet_20201031 import OCTDataSet_20201031
from OCTUnetTongren import OCTUnetTongren
from OCTUnetJHU import OCTUnetJHU
from OCTUnetSurfaceLayerJHU import OCTUnetSurfaceLayerJHU
from IVUSUnet import IVUSUnet
from SurfacesUnet_20201031 import SurfacesUnet_20201031
from OCTOptimization import *
from OCTTransform import OCTDataTransform

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

    trainTransform = OCTDataTransform(prob=hps.augmentProb, noiseStd=hps.gaussianNoiseStd, saltPepperRate=hps.saltPepperRate, saltRate=hps.saltRate, rotation=hps.rotation)
    validationTransform = trainTransform
    # validation supporting data augmentation benefits both learning rate decaying and generalization.

    trainData = eval(hps.dataSetClass)(trainImagesPath, trainIDPath, trainLabelsPath,  transform=trainTransform, hps=hps)
    validationData = eval(hps.dataSetClass)(validationImagesPath, validationIDPath, validationLabelsPath,  transform=validationTransform,  hps=hps)

    # construct network
    net = eval(hps.network)(hps=hps)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=hps.device)

    # optimizer and lr greatly affect result performance
    # optimizer = optim.Adam(net.parameters(), lr=hps.learningRate, weight_decay=0)
    optimizer = eval(hps.optimizer)(net.parameters(), lr=hps.learningRate, weight_decay=hps.weightDecay)
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=hps.patience, min_lr=1e-8, threshold=0.02, threshold_mode='rel')
    net.setLrScheduler(lrScheduler)

    # Load network
    netMgr = NetMgr(net, hps.netPath, hps.device)
    netMgr.loadNet(hps.mode)

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

    for epoch in range(initialEpoch, epochs):
        random.seed()
        net.m_epoch = epoch
        net.setStatus(hps.mode)

        net.train()
        trBatch = 0
        trLoss = 0.0
        for batchData in data.DataLoader(trainData, batch_size=hps.batchSize, shuffle=True, num_workers=0):
            trBatch += 1
            forwardOutput = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'])
            if hps.outputLatent:
                _S, loss, latentVector = forwardOutput
            else:
                _S, loss = forwardOutput

            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(hps.device))
            optimizer.step()
            trLoss += loss


        trLoss = trLoss / trBatch

        net.eval()
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0
            net.setStatus("validation")
            for batchData in data.DataLoader(validationData, batch_size=hps.batchSize, shuffle=False,
                                                              num_workers=0):
                validBatch += 1
                # S is surface location in (B,S,W) dimension, the predicted Mu
                forwardOutput = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'])
                if hps.outputLatent:
                    S, loss, latentVector = forwardOutput
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
            net.updateRunParameter("learningRate", optimizer.param_groups[0]['lr'])
            preValidLoss = validLoss
            preErrorMean = muError
            netMgr.saveNet(hps.netPath)




    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()
