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
    with open(configFile) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    experimentName = getStemName(configFile, removedSuffix=".yaml")
    print(f"Experiment: {experimentName}")

    dataDir = cfg["dataDir"]
    K = cfg["K_Folds"]
    k = cfg["fold_k"]

    groundTruthInteger = cfg["groundTruthInteger"]

    sigma = cfg["sigma"]  # for gausssian ground truth
    device = eval(cfg["device"])  # convert string to class object.
    batchSize = cfg["batchSize"]

    network = cfg["network"]#
    inputHight= cfg["inputHight"] # 192
    inputWidth= cfg["inputWidth"] #1060  # rawImageWidth +2 *lacingWidth
    scaleNumerator= cfg["scaleNumerator"] #2
    scaleDenominator= cfg["scaleDenominator"] #3
    inputChannels= cfg["inputChannels"] #1
    nLayers= cfg["nLayers"] #7
    numSurfaces = cfg["numSurfaces"]
    numStartFilters = cfg["startFilters"]  # the num of filter in first layer of Unet
    gradChannels= cfg["gradChannels"]
    gradWeight = cfg["gradWeight"]

    slicesPerPatient = cfg["slicesPerPatient"] # 31
    hPixelSize = cfg["hPixelSize"] #  3.870  # unit: micrometer, in y/height direction

    augmentProb = cfg["augmentProb"]
    gaussianNoiseStd = cfg["gaussianNoiseStd"]  # gausssian nosie std with mean =0
    # for salt-pepper noise
    saltPepperRate= cfg["saltPepperRate"]   # rate = (salt+pepper)/allPixels
    saltRate= cfg["saltRate"]  # saltRate = salt/(salt+pepper)
    lacingWidth = cfg["lacingWidth"]
    if "IVUS" in dataDir:
        rotation = cfg["rotation"]
    else:
        rotation = False


    netPath = cfg["netPath"] + "/" + network + "/" + experimentName
    loadNetPath = cfg['loadNetPath']
    if "" != loadNetPath:
        netPath = loadNetPath

    lossFunc0 = cfg["lossFunc0"] # "nn.KLDivLoss(reduction='batchmean').to(device)"
    lossFunc0Epochs = cfg["lossFunc0Epochs"] #  the epoch number of using lossFunc0
    lossFunc1 = cfg["lossFunc1"] #  "nn.SmoothL1Loss().to(device)"
    lossFunc1Epochs = cfg["lossFunc1Epochs"] # the epoch number of using lossFunc1

    # Proximal IPM Optimization
    useProxialIPM = cfg['useProxialIPM']
    if useProxialIPM:
        learningStepIPM =cfg['learningStepIPM'] # 0.1
        maxIterationIPM =cfg['maxIterationIPM'] # : 50
        criterionIPM = cfg['criterionIPM']

    useDynamicProgramming = cfg['useDynamicProgramming']
    usePrimalDualIPM = cfg['usePrimalDualIPM']
    useCEReplaceKLDiv = cfg['useCEReplaceKLDiv']
    useLayerDice = cfg['useLayerDice']
    useReferSurfaceFromLayer = cfg['useReferSurfaceFromLayer']
    useLayerCE = cfg['useLayerCE']
    useSmoothSurfaceLoss = cfg['useSmoothSurfaceLoss']
    useWeightedDivLoss = cfg['useWeightedDivLoss']

    if -1==k and 0==K:  # do not use cross validation
        trainImagesPath = os.path.join(dataDir, "training", f"images.npy")
        trainLabelsPath = os.path.join(dataDir, "training", f"surfaces.npy")
        trainIDPath = os.path.join(dataDir, "training", f"patientID.json")

        validationImagesPath = os.path.join(dataDir, "test", f"images.npy")
        validationLabelsPath = os.path.join(dataDir, "test", f"surfaces.npy")
        validationIDPath = os.path.join(dataDir, "test", f"patientID.json")
    else:  # use cross validation
        trainImagesPath = os.path.join(dataDir,"training", f"images_CV{k:d}.npy")
        trainLabelsPath  = os.path.join(dataDir,"training", f"surfaces_CV{k:d}.npy")
        trainIDPath     = os.path.join(dataDir,"training", f"patientID_CV{k:d}.json")

        validationImagesPath = os.path.join(dataDir,"validation", f"images_CV{k:d}.npy")
        validationLabelsPath = os.path.join(dataDir,"validation", f"surfaces_CV{k:d}.npy")
        validationIDPath    = os.path.join(dataDir,"validation", f"patientID_CV{k:d}.json")

    trainTransform = OCTDataTransform(prob=augmentProb, noiseStd=gaussianNoiseStd, saltPepperRate=saltPepperRate, saltRate=saltRate, rotation=rotation)
    validationTransform = trainTransform
    # validation supporting data augmentation benefits both learning rate decaying and generalization.

    trainData = OCTDataSet(trainImagesPath, trainLabelsPath, trainIDPath, transform=trainTransform, device=device, sigma=sigma, lacingWidth=lacingWidth,
                           TTA=False, TTA_Degree=0, scaleNumerator=scaleNumerator, scaleDenominator=scaleDenominator,
                           gradChannels=gradChannels)
    validationData = OCTDataSet(validationImagesPath, validationLabelsPath, validationIDPath, transform=validationTransform, device=device, sigma=sigma,
                                lacingWidth=lacingWidth, TTA=False, TTA_Degree=0, scaleNumerator=scaleNumerator, scaleDenominator=scaleDenominator,
                                gradChannels=gradChannels)

    # construct network
    net = eval(network)(inputHight, inputWidth, inputChannels=inputChannels, nLayers=nLayers, numSurfaces=numSurfaces, N=numStartFilters)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device=device)

    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=0)
    net.setOptimizer(optimizer)
    lrScheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-8, threshold=0.02, threshold_mode='rel')

    # KLDivLoss is for Guassuian Ground truth for Unet
    loss0 = eval(lossFunc0) #nn.KLDivLoss(reduction='batchmean').to(device)  # the input given is expected to contain log-probabilities
    net.appendLossFunc(loss0, weight=1.0, epochs=lossFunc0Epochs)
    loss1 = eval(lossFunc1)
    net.appendLossFunc(loss1, weight=1.0, epochs=lossFunc1Epochs)

    # Load network
    if os.path.exists(netPath) and len(getFilesList(netPath, ".pt")) >= 2 :
        netMgr = NetMgr(net, netPath, device)
        netMgr.loadNet("train")
        print(f"Network load from  {netPath}")
    else:
        netMgr = NetMgr(net, netPath, device)
        print(f"Net starts training from scratch, and save at {netPath}")

    # according config file to update config parameter
    net.updateRunParameter('useProxialIPM', useProxialIPM)
    if useProxialIPM:
        net.updateRunParameter("learningStepIPM", learningStepIPM)
        net.updateRunParameter("maxIterationIPM", maxIterationIPM)
        net.updateRunParameter("criterion", criterionIPM)

    net.updateRunParameter("useDynamicProgramming", useDynamicProgramming)
    net.updateRunParameter("usePrimalDualIPM", usePrimalDualIPM)
    net.updateRunParameter("useCEReplaceKLDiv", useCEReplaceKLDiv)
    net.updateRunParameter("useLayerDice", useLayerDice)
    net.updateRunParameter("useReferSurfaceFromLayer", useReferSurfaceFromLayer)
    net.updateRunParameter("useSmoothSurfaceLoss", useSmoothSurfaceLoss)
    net.updateRunParameter("gradWeight", gradWeight)
    net.updateRunParameter("useWeightedDivLoss", useWeightedDivLoss)
    net.updateRunParameter("useLayerCE", useLayerCE)

    logDir = dataDir + "/log/" + network + "/" + experimentName
    if not os.path.exists(logDir):
        os.makedirs(logDir)  # recursive dir creation
    writer = SummaryWriter(log_dir=logDir)

    # train
    epochs = 1360000
    preTrainingLoss = 999999.0
    preValidLoss = net.getRunParameter("validationLoss") if "validationLoss" in net.m_runParametersDict else 2041  # float 16 has maxvalue: 2048
    preErrorMean = net.getRunParameter("errorMean") if "errorMean" in net.m_runParametersDict else 3.3
    if net.training:
        initialEpoch = net.getRunParameter("epoch") if "epoch" in net.m_runParametersDict else 0
    else:
        initialEpoch = 0

    for epoch in range(initialEpoch, epochs):
        random.seed()
        net.m_epoch = epoch

        net.train()
        trBatch = 0
        trLoss = 0.0
        for batchData in data.DataLoader(trainData, batch_size=batchSize, shuffle=True, num_workers=0):
            trBatch += 1
            _S, loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'])
            optimizer.zero_grad()
            loss.backward(gradient=torch.ones(loss.shape).to(device))
            optimizer.step()
            trLoss += float(loss)

        trLoss = trLoss / trBatch
        #lrScheduler.step(trLoss)

        net.eval()
        with torch.no_grad():
            validBatch = 0  # valid means validation
            validLoss = 0.0
            for batchData in data.DataLoader(validationData, batch_size=batchSize, shuffle=False,
                                                              num_workers=0):
                validBatch += 1
                # S is surface location in (B,S,W) dimension, the predicted Mu
                S, loss = net.forward(batchData['images'], gaussianGTs=batchData['gaussianGTs'], GTs = batchData['GTs'], layerGTs=batchData['layers'])
                validLoss += float(loss)
                validOutputs = torch.cat((validOutputs, S)) if validBatch != 1 else S
                validGts = torch.cat((validGts, batchData['GTs'])) if validBatch != 1 else batchData['GTs'] # Not Gaussian GTs
                validIDs = validIDs + batchData['IDs'] if validBatch != 1 else batchData['IDs']  # for future output predict images

            validLoss = validLoss / validBatch
            if groundTruthInteger:
                validOutputs = (validOutputs+0.5).int() # as ground truth are integer, make the output also integers.
            # Error Std and mean
            stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(validOutputs, validGts,
                                                                                                 slicesPerPatient=slicesPerPatient,
                                                                                                 hPixelSize=hPixelSize)
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
            netMgr.saveNet(netPath)

        if trLoss < preTrainingLoss:
            preTrainingLoss = trLoss
            netMgr.saveRealTimeNet(netPath)


    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()
