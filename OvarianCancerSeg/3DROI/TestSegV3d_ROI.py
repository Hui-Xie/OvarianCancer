# train Seg 3d V model
import datetime

from OvarianCancerSeg.OCDataSegSet import *
from utilities.FilesUtilities import *
from utilities.MeasureUtilities import *
from OvarianCancerSeg.trial.SegV3DModel import SegV3DModel
from OvarianCancerSeg.OCDataTransform import *
from framework.NetMgr import NetMgr

def printUsage(argv):
    print("============Test Seg 3D VNet for ROI around primary Cancer =============")
    print("Usage:")
    print(argv[0],
          "<netSavedPath> <predictOutputDir> <fullPathOfData>  <fullPathOfLabel> <k>  <GPUID>")
    print("where: \n"
          "       netSavedPath must be specific network directory.\n"
          "       k=[0, K), the k-th fold in the K-fold cross validation.\n"
          "       GPUID:    one of 0,1,2,3, the specific GPU ID List, separated by comma\n")


def main():
    if len(sys.argv) != 7:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    netPath = sys.argv[1]
    predictOutputDir = sys.argv[2]
    dataInputsPath = sys.argv[3]
    groundTruthPath = sys.argv[4]
    k = int(sys.argv[5])
    GPUID = int(sys.argv[6])  # choices: 0,1,2,3 for lab server.
    useConsistencyLoss = False

    if not os.path.exists(predictOutputDir):
        os.mkdir(predictOutputDir)

    print(f'Program ID:  {os.getpid()}\n')
    print(f'Program commands: {sys.argv}')
    print(f'.........')

    inputSuffix = ".npy"
    K_fold = 1 #
    batchSize = 1
    print(f"batchSize = {batchSize}")
    numWorkers = 0

    device = torch.device(f"cuda:{GPUID}" if torch.cuda.is_available() else "cpu")

    timeStr = getStemName(netPath)
    if timeStr == "Best":
        timeStr = getStemName(netPath.replace("/Best", ""))

    dataPartitions = OVDataSegPartition(dataInputsPath, groundTruthPath, inputSuffix, K_fold, k)

    if 0 == K_fold or 1 == K_fold:
        allTransform = OCDataLabelTransform(0)
        allData = OVDataSegSet('all', dataPartitions, transform=allTransform)
    else:
        validationTransform = OCDataLabelTransform(0)
        testTransform = OCDataLabelTransform(0)
        validationData = OVDataSegSet('validation', dataPartitions, transform=validationTransform)
        testData = OVDataSegSet('test', dataPartitions, transform=testTransform)

    net = SegV3DModel(useConsistencyLoss=useConsistencyLoss)
    # Important:
    # If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    net.to(device)

    # Load network
    netMgr = NetMgr(net, netPath, device)
    netMgr.loadNet("test")

    patientDiceMap = {}
    net.eval()

    if 0 == K_fold or 1 == K_fold:
        # ===============All data test =======================
        # with weak annotation ground truth.
        with torch.no_grad():
            for inputs, labels, patientIDs in data.DataLoader(allData, batch_size=batchSize, shuffle=False,
                                                              num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gts = labels.to(device, dtype=torch.float)  # return a copy
                gts = (gts > 0).long()  # not discriminate all non-zero labels.

                outputs, _ = net.forward(inputs, gts)

                # compute dice
                gtsShape = gts.shape
                for i in range(gtsShape[0]):
                    output = torch.argmax(outputs[i,], dim=0)
                    gt = gts[i,]
                    dice = weakTensorDice(output, gt)
                    filename = os.path.join(predictOutputDir, patientIDs[i] + ".npy")
                    np.save(filename, output.cpu().numpy())
                    patientDiceMap[patientIDs[i]] = dice


    else:
        # ================Validation===============
        with torch.no_grad():
            for inputs, labels, patientIDs in data.DataLoader(validationData, batch_size=batchSize, shuffle=False,
                                                              num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gts = labels.to(device, dtype=torch.float)  # return a copy
                gts = (gts > 0).long()  # not discriminate all non-zero labels.

                outputs, _ = net.forward(inputs, gts)

                # compute dice
                gtsShape = gts.shape
                for i in range(gtsShape[0]):
                    output = torch.argmax(outputs[i,], dim=0)
                    gt = gts[i,]
                    dice = tensorDice(output, gt)
                    filename = os.path.join(predictOutputDir, patientIDs[i]+".npy")
                    np.save(filename, output.cpu().numpy())
                    patientDiceMap[patientIDs[i]]= dice

        # ================Independent Test===============
        with torch.no_grad():
            for inputs, labels, patientIDs in data.DataLoader(testData, batch_size=batchSize, shuffle=False,
                                                              num_workers=numWorkers):
                inputs = inputs.to(device, dtype=torch.float)
                gts = labels.to(device, dtype=torch.float)  # return a copy
                gts = (gts > 0).long()  # not discriminate all non-zero labels.

                outputs, _ = net.forward(inputs, gts)

                # compute dice
                gtsShape = gts.shape
                for i in range(gtsShape[0]):
                    output = torch.argmax(outputs[i,], dim=0)
                    gt = gts[i,]
                    dice = tensorDice(output, gt)
                    filename = os.path.join(predictOutputDir, patientIDs[i] + ".npy")
                    np.save(filename, output.cpu().numpy())
                    patientDiceMap[patientIDs[i]]= dice

    # statistic dic
    numPatients = len(patientDiceMap)
    diceValues = np.asarray(list(patientDiceMap.values()), dtype=np.float32)
    print(f"total {numPatients} patients for test")
    print(f"min dice = {diceValues.min()}")
    print(f"max dice = {diceValues.max()}")
    print(f"mean dice = {diceValues.mean()}")
    print(f"dice std  = {diceValues.std()}")

    # output dice dictionary
    jsonData = json.dumps(patientDiceMap)
    outfile = os.path.join(predictOutputDir, "patientDice.json")
    f = open(outfile, "w")
    f.write(jsonData)
    f.close()

    torch.cuda.empty_cache()
    print(f"\n\n=============END of Test of SegV3d ROI Model =================")
    print(f'Program ID {os.getpid()}  exits.\n')
    curTime = datetime.datetime.now()
    print(f'\nProgram Ending Time: {str(curTime)}')


if __name__ == "__main__":
    main()
