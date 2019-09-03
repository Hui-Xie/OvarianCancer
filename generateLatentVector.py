# base on the script of trainResNeXtVNet,
# generate all latent vectors for a trained ResNeXt V Model

import sys
import datetime
import torch.nn as nn
from torch.utils import data
import logging

from OCDataSegSet import *
from FilesUtilities import *
from MeasureUtilities import *
from ResNeXtVNet import ResNeXtVNet
from NetMgr import NetMgr

logNotes = r'''
Major program changes: 
     1  basing trained ResNeXt V model, generate all latent vector for segmented and unsegmented data

Discarded changes:                  

Input data: maximum size 231*251*251 (zyx) of 3D numpy array with spacing size(3*2*2), without data augmentation
            total 220 patients data which includes segmented and unsegmented data. 
          '''

def printUsage(argv):
    print("============Generate all latent vectors =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath>  <fullPathOfData>  <fullPathOfLatentDir>  <GPUID_List>")
    print("where: \n"
          "       fullPathOflatentDir: the output director for latent vector;\n"
          "       GPUIDList: 3,2,1,0 the specific GPU ID List, separated by comma.\n")


def main():
    if len(sys.argv) != 5:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    dataInputsPath = sys.argv[2]
    outputPath = sys.argv[3]
    GPUIDList = sys.argv[4].split(',')  # choices: 0,1,2,3 for lab server.
    GPUIDList = [int(x) for x in GPUIDList]

    # avoid overwriting latent vector dir
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    else:
        if len(os.listdir(outputPath)) != 0:
            print (f"{outputPath} contains files. You need to delete or move them first.\n")
            return -1

    # ===========debug==================
    useDataParallel = True if len(GPUIDList) > 1 else False
    # ===========debug==================

    print(f'Program ID:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'.........')

    inputSuffix = ".npy"
    batchSize = 22 * len(GPUIDList)
    print(f"batchSize = {batchSize}")
    numWorkers = 0

    device = torch.device(f"cuda:{GPUIDList[0]}" if torch.cuda.is_available() else "cpu")

    timeStr = getStemName(netPath)

    logFile = os.path.join(outputPath, f'latentLog_{timeStr}.txt')
    print(f'log is in {logFile}')
    logging.basicConfig(filename=logFile, filemode='a+', level=logging.INFO, format='%(message)s')

    logging.info(f'Program command: \n {sys.argv}')

    dataPartitions = OVDataSegPartition(dataInputsPath, inputSuffix=inputSuffix)
    fullData = OVDataSegSet('fulldata', dataPartitions)

    net = ResNeXtVNet()
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    # Load network
    netMgr = NetMgr(net, netPath, device)

    if 2 == len(getFilesList(netPath, ".pt")):
        netMgr.loadNet("test")
    else:
        logging.info(net.getParametersScale())

    if useDataParallel:
        net = nn.DataParallel(net, device_ids=GPUIDList, output_device=device)

    # ================ Generate all latent vectors ===============
    net.eval()
    with torch.no_grad():
        for inputs,patientIDs in data.DataLoader(fullData, batch_size=batchSize, shuffle=False, num_workers=numWorkers):
            inputs = inputs.to(device, dtype=torch.float)

            outputs = net.forward(inputs, halfForward=True)
            for i in range(outputs.shape[0]):
                output = outputs[i].cpu().detach().numpy()
                if 0 == np.std(output):
                    logging.info(f"patientID {patientIDs[i]} has latent vector of full zero.")
                np.save(os.path.join(outputPath, patientIDs[i] + ".npy"), output)

    torch.cuda.empty_cache()
    logging.info(f"\n\n============= END of Generating latent Vectoors =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    logging.info(f'\nProgram Ending Time: {str(curTime)}')

if __name__ == "__main__":
    main()
