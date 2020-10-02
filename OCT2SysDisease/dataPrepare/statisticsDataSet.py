# statistics dataset

gtPath= "/home/hxie1/data/BES_3K/GTs/BESClinicalGT.csv"

import sys
sys.path.append(".")
from readClinicalGT import readBESClinicalCsv

def statisticsData(dataSetIDPath, key="", valueType=None):
    '''

    :param dataSetIDPath:
    :param key:
    :param valueType: binary, number,12binary
    :return:
    '''
    gtDict = readBESClinicalCsv(gtPath)

    with open(dataSetIDPath,'r') as f:
        IDList = f.readlines()
    IDList = [item[0:-1] for item in IDList] # erase '\n'


    minV = -1000
    maxV = -1000
    avgV = 0
    b0Count = 0
    b1Count = 0
    N = 0

    for ID in IDList:
        value = gtDict[ID][key]
        if -100 == value:
            continue
        N +=1
        if valueType=="binary":
            if 0 == value:
                b0Count +=1
            elif 1 == value:
                b1Count +=1
            else:
                print(f"value= {value} don't match binary type at ID {ID}")
                assert False

        elif valueType == "number":
            if 1 ==N:
                minV = value
                maxV = value
                avgV = value
            else:
                minV = value if value < minV else minV
                maxV = value if value > maxV else maxV
                avgV +=value
        elif valueType == "12binary":
            if 1 == value:
                b0Count += 1
            elif 2 == value:
                b1Count += 1
            else:
                print(f"value= {value} don't match 12binary type at ID {ID}")
                assert False
        else:
            print("valueType error")
            assert False

    # print result
    print(f"total {len(IDList)} raw IDs in file {dataSetIDPath}")
    print(f"values for {key} have {N} records")
    if (valueType == "binary"):
        print(f"0 rate: {b0Count/N}; 1 rate: {b1Count/N}")
    elif (valueType == "12binary"):
        print(f"1 rate: {b0Count/N}; 2 rate: {b1Count/N}")
    elif valueType == "number":
        avgV /=N
        print(f"min={minV}; mean={avgV}; max={maxV}")
    else:
        print("valueType error")
        assert False

def printUsage(argv):
    print("============ statistics specific key in a ID dataset  =============")
    print("Usage:")
    print(argv[0], " ID_path  keyName  <binary : 12binary : number>")


def main():

    if len(sys.argv) != 4:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1


    dataSetIDPath = sys.argv[1]
    key = sys.argv[2]
    valueType = sys.argv[3]
    statisticsData(dataSetIDPath, key=key, valueType=valueType)


if __name__ == "__main__":
    main()
