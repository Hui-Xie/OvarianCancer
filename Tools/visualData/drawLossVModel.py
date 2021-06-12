#  draw loss or accuracy curver

import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utilities.FilesUtilities import *
import os

def printUsage(argv):
    print("============Draw Loss or accuracy Curve =============")
    print("Usage:")
    print(argv[0], " <logfile>  <deleteTopNRows> ")
    print("where, if delteTopNRows is 2 as example, array will first delete the top 0, 1 rows before drawing.")

def main():
    if len(sys.argv) != 3:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    logFile = sys.argv[1]
    deleteTopN = int(sys.argv[2])
    experiment = getStemName(logFile, ".txt")
    fullPathLogFile = os.path.abspath(logFile)
    dirName = os.path.dirname(fullPathLogFile)

    tableTitle = "************** Table of Training Log **************\n"

    # read data
    with open(logFile) as file:
        data = file.read()
        pos = data.find(tableTitle)
        data = data[pos+len(tableTitle):]
        lines = data.splitlines()
        lines.pop(0)   #erase tabel head.
        array = np.zeros((len(lines),8),dtype=float)

        countRow = 0
        for line in lines:
            line = line.replace('\t\t','\t')
            row = line.split('\t')
            if len(row) != 8:
                break
            else:
                array[countRow,] = np.asarray(row)
                countRow +=1
        array = array[:countRow,]

    delRowList = [x for x in range(deleteTopN)]
    array = np.delete(array, delRowList, 0)

    # draw curve
    colsLoss = [2,4,6]
    colsDice = [3,5,7]

    # draw Loss
    f1 = plt.figure(1)
    plt.plot(array[:,0], array[:,colsLoss])
    plt.legend(('Training','Validation', 'Test'))
    plt.title(f"Loss in {experiment}")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(dirName, f"Loss_{experiment}.png"))

    # draw Accuracy
    f2 = plt.figure(2)
    plt.plot(array[:, 0], array[:, colsDice])
    plt.legend(('Training', 'Validation', 'Test'))
    plt.title(f"Accuracy in {experiment}")
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.savefig(os.path.join(dirName,f"Dice_{experiment}.png"))

    # draw learning Rate
    f3 = plt.figure(3)
    plt.plot(array[:, 0], array[:, 1])
    plt.legend(('LearningRate'))
    plt.title(f"LearningRate in {experiment}")
    plt.xlabel('Epoch')
    plt.ylabel('LearningRate')
    plt.savefig(os.path.join(dirName, f"LearningRate_{experiment}.png"))

    plt.show()
    return

if __name__ == "__main__":
    main()
