# support various input size

# need python package:  pillow(6.2.1), opencv, pytorch, torchvision, tensorboard

import sys
import random

from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

sys.path.append("..")
from network.OCTDataSet_A import OCTDataSet_A
from network.OCTOptimization import *
from network.OCTTransform import *

sys.path.append(".")
from SoftSeparationNet_A import SoftSeparationNet_A

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

            if hps.status == "trainLambda" and hps.trainLambdaUsingValidation:
                trainImagesPath = validationImagesPath
                trainLabelsPath = validationLabelsPath
                trainIDPath     = validationIDPath
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

    trainData = OCTDataSet_A(trainImagesPath, trainIDPath, trainLabelsPath,  transform=trainTransform, hps=hps)
    validationData = OCTDataSet_A(validationImagesPath, validationIDPath, validationLabelsPath,  transform=validationTransform,  hps=hps)

    # construct network
    net = eval(hps.network)(hps=hps)
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
        net.setStatus("training")

        net.train()
        trBatch = 0
        trLoss = 0.0
        for batchData in data.DataLoader(trainData, batch_size=hps.batchSize, shuffle=True, num_workers=0, drop_last=True):
            trBatch += 1
            S, loss = net.forward(batchData['images'], batchData['imageYX'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
            net.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(loss.device))
            net.optimizerStep()
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
                S, loss  = net.forward(batchData['images'], batchData['imageYX'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'], riftGTs=batchData['riftWidth'])
                validLoss += loss
                validOutputs = torch.cat((validOutputs, S)) if validBatch != 1 else S
                validGts = torch.cat((validGts, batchData['GTs'])) if validBatch != 1 else batchData['GTs'] # Not Gaussian GTs
                validIDs = validIDs + batchData['IDs'] if validBatch != 1 else batchData['IDs']  # for future output predict images

            validLoss = validLoss / validBatch
            if hps.groundTruthInteger:
                validOutputs = (validOutputs+0.5).int() # as ground truth are integer, make the output also integers.
            # print(f"epoch:{epoch}; validLoss ={validLoss}\n")

            goodBScansInGtOrder =None
            stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(validOutputs, validGts,
                                                                                                 slicesPerPatient=hps.slicesPerPatient,
                                                                                                 hPixelSize=hps.hPixelSize,
                                                                                                 goodBScansInGtOrder=goodBScansInGtOrder)

        net.lrSchedulerStep(validLoss)
        # debug
        # print(f"epoch {epoch} ends...")  # for smoke debug

        writer.add_scalars('Loss', {"train": trLoss, "validation": validLoss}, epoch)
        writer.add_scalar('ValidationError/mean', muError, epoch)
        writer.add_scalar('ValidationError/std', stdError, epoch)
        writer.add_scalars('ValidationError/muSurface', convertTensor2Dict(muSurfaceError), epoch)
        writer.add_scalars('ValidationError/stdSurface', convertTensor2Dict(stdSurfaceError), epoch)
        if hps.status == "trainLambda":
            writer.add_scalar('learningRateLambda', net.getLearningRate("lambdaModule"), epoch)
        else:
            pass

        if validLoss < preValidLoss or muError < preErrorMean:
            net.updateRunParameter("validationLoss", validLoss)
            net.updateRunParameter("epoch", net.m_epoch)
            net.updateRunParameter("errorMean", muError)
            net.updateRunParameter("learningRate_Surface", net.getLearningRate("surfaceSubnet"))
            net.updateRunParameter("learningRate_Rift", net.getLearningRate("riftSubnet"))
            preValidLoss = validLoss
            preErrorMean = muError
            net.saveNet()

        #debug
        print("passed smoke test")
        break

    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()
