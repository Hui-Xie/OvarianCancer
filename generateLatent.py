# Generate all Latent Vector of V model
# input: the uniform nrrd images and labels directories
# model: the trained U-Net
# output: all latent vector named with patient ID

import sys
import datetime
import random
import torch
import torch.nn as nn
import logging
import os

from LatentDataMgr import LatentDataMgr
from SegV2DModel import SegV2DModel
from SegV3DModel import SegV3DModel
from NetMgr import NetMgr
from CustomizedLoss import FocalCELoss, BoundaryLoss


# you may need to change the file name and log Notes below for every training.
generateLatentLog = r'''/home/hxie1/Projects/OvarianCancer/trainLog/latentLog_20190520.txt'''
logNotes = r'''
Major program changes of V model: 
                       merge train and test dataset;
                       for primary and metastases 3 classes classification
                       use conv-BN-Relu order;
                       use Dense module
                       Use ResPath
                       the nunmber of filters in 1st layer = 96

            '''

logging.basicConfig(filename=generateLatentLog, filemode='a+', level=logging.INFO, format='%(message)s')


def printUsage(argv):
    print("============Generate all latent vector of segmentattion V model=============")
    print("Usage:")
    print(argv[0], "<netSavedPath> <fullPathOfTrainImages>  <fullPathOfTrainLabels>  <2D|3D> <labelTuple>")
    print("eg. labelTuple: (0,1,2,3), or (0,1), (0,2)")


def main():
    if len(sys.argv) != 6:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    print(f'Program ID {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'Latent log is in {generateLatentLog}')
    print(f'.........')

    logging.info(f'Program ID {os.getpid()}\n')
    logging.info(f'Program command: \n {sys.argv}')
    logging.info(logNotes)

    heightVolume =  51 # original uniform min image height is 147

    curTime = datetime.datetime.now()
    logging.info(f'\nProgram starting Time: {str(curTime)}')

    netPath = sys.argv[1]
    imagesPath = sys.argv[2]
    labelsPath = sys.argv[3]
    is2DInput = True if sys.argv[4] == "2D" else False
    labelTuple = eval(sys.argv[5])
    K = len(labelTuple)

    logging.info(f"Info: netPath = {netPath}\n")

    mergeTrainTestData = True

    dataMgr = LatentDataMgr(imagesPath, labelsPath, logInfoFun=logging.info)
    dataMgr.setRemainedLabel(3, labelTuple)

    # ===========debug==================
    dataMgr.setOneSampleTraining(False)  # for debug
    useDataParallel = True  # for debug
    outputTrainDice = True
    # ===========debug==================

    dataMgr.expandImagesDir(dataMgr.getTestDirs()[0])

    if is2DInput:
        logging.info(f"Info: program uses 2D input.")
        dataMgr.setDataSize(8, 1, 281, 281, K,"TrainData")  # batchSize, depth, height, width, k, # do not consider lymph node with label 3
        net = SegV2DModel(96, K)
    else:
        logging.info(f"Info: program uses 3D input.")
        dataMgr.setDataSize(4, 21, 281, 281, K, "TrainData")  # batchSize, depth, height, width, k
        net = SegV3DModel(K)

    # Load network
    netMgr = NetMgr(net, netPath)
    bestTestDiceList = [0] * K
    if 2 == len(dataMgr.getFilesList(netPath, ".pt")):
        netMgr.loadNet("test")  # True for train
        logging.info(f'Program loads net from {netPath}.')
        bestTestDiceList = netMgr.loadBestTestDice(K)
        logging.info(f'Current best test dice: {bestTestDiceList}')
    else:
        logging.info(f"Network trains from scratch.")

    logging.info(net.getParametersScale())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
    if useDataParallel:
        nGPU = torch.cuda.device_count()
        if nGPU > 1:
            logging.info(f'Info: program will use {nGPU} GPUs.')
            net = nn.DataParallel(net, device_ids=[0, 1, 2, 3], output_device=device)

    net.eval()
    for image in dataMgr.m_imagesList:
        patientID = dataMgr.getStemName(image, "_CT.nrrd")
        with torch.no_grad():
            assembleLatent = torch.tensor([]).to(device)
            for inputs in dataMgr.sectionGenerator(image,heightVolume):
                inputs = torch.from_numpy(inputs)
                inputs = inputs.to(device, dtype=torch.float)
                if useDataParallel:
                    latentV = net.module.halfForward(inputs)
                else:
                    latentV = net.halfForward(inputs)
                assembleLatent = torch.cat((assembleLatent, latentV))
            assembleLatent = assembleLatent.cpu().numpy().astype(float)
        dataMgr.saveLatentV(assembleLatent, patientID)

    torch.cuda.empty_cache()
    logging.info(f"=============END of Training of Ovarian Cancer Segmentation V Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    print(f"=============END of Training of Ovarian Cancer Segmentation V Model =================")


if __name__ == "__main__":
    main()
