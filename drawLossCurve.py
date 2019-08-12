#  draw loss or accuracy curver

import sys
import numpy as np
import matplotlib.pyplot as plt
from FilesUtilities import *

def printUsage(argv):
    print("============Draw Loss or accuracy Curve =============")
    print("Usage:")
    print(argv[0], "logfile")

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    logFile = sys.argv[1]
    experiment = getStemName(logFile, ".txt")

    tableHead = "Epoch	LearningRate		TrLoss	Accura	TPR_r	TNR_r		VaLoss	Accura	TPR_r	TNR_r		TeLoss	Accura	TPR_r	TNR_r\n"

    # read data
    with open(logFile) as file:
        data = file.read()
        pos = data.find(tableHead)
        data = data[pos+len(tableHead):]
        lines = data.splitlines()
        array = np.zeros((len(lines),14),dtype=np.float32)

        countRow = 0
        for line in lines:
            line = line.replace('\t\t','\t')
            row = line.split('\t')
            if len(row) != 14:
                break
            else:
                array[countRow,] = np.asarray(row)
                countRow +=1
        array = array[:countRow,]

    # draw curve
    colsLoss = [2,6,10]
    colsAccuracy = [3,7,11]

    # draw Loss
    f1 = plt.figure(1)
    plt.plot(array[:,0], array[:,colsLoss])
    plt.legend(('Training','Validation', 'Test'))
    plt.title(f"Loss in {experiment}")


    # draw Accuracy
    f2 = plt.figure(2)
    plt.plot(array[:, 0], array[:, colsAccuracy])
    plt.legend(('Training', 'Validation', 'Test'))
    plt.title(f"Accuracy in {experiment}")

    plt.show()
    return

if __name__ == "__main__":
    main()