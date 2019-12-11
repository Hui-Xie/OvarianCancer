# generate all latent vectors for a trained V Model
# output is saved as one numpy file for one sample

import datetime
import logging

from OCDataSet import *
from FilesUtilities import *
from SegV3DModel import *
from NetMgr import *

def printUsage(argv):
    print("============Generate all latent vectors =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath>  <fullPathOfData> <fullPathResponse> <fullPathOfLatentDir>  <GPUID>")
    print("where: \n"
          "       fullPathOflatentDir: the output director for latent vector;\n"
          "       GPUID: one of 3,2,1,0, the specific GPU ID.\n")


def main():
    if len(sys.argv) != 6:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    timeStr = getStemName(netPath)
    if timeStr == "Best":
        timeStr = getStemName(netPath.replace("/Best", ""))

    dataInputsPath = sys.argv[2]
    responseFile = sys.argv[3]
    outputPath = os.path.join(sys.argv[4], f"latent_{timeStr}")
    GPUID = int(sys.argv[5])

    # avoid overwriting latent vector dir
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    else:
        if len(os.listdir(outputPath)) != 0:
            print (f"{outputPath} contains files. You need to delete or move them first.\n")
            return -1

    inputSuffix = ".npy"
    batchSize = 1
    print(f"batchSize = {batchSize}")

    device = torch.device(f"cuda:{GPUID}" if torch.cuda.is_available() else "cpu")

    dataPartitions = OVDataPartition(dataInputsPath, responseFile, inputSuffix, K_fold=0, k=0, logInfoFun=print)

    allData = OVDataSet('all', dataPartitions, transform=None, logInfoFun=logging.info)

    net = SegV3DModel()
    net.to(device)
    # Load network
    netMgr = NetMgr(net, netPath, device)
    netMgr.loadNet("test")

    # ================Validation===============
    net.train()
    with torch.no_grad():
        for inputs, labels, patientIDs in data.DataLoader(allData, batch_size=batchSize, shuffle=False, num_workers=0):
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.int()
            outputs = net.forward(inputs, gts=None, halfForward=True)
            for i in range(outputs.shape[0]):
                output = outputs[i].detach().cpu().numpy()
                if 0 == np.std(output):
                    logging.info(f"patientID {patientIDs[i]} has latent vector of full zero.")
                np.save(os.path.join(outputPath, patientIDs[i] + ".npy"), output)


    torch.cuda.empty_cache()
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    print(f'\nProgram Ending Time: {str(curTime)}')

if __name__ == "__main__":
    main()
