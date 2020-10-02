# check whether the OCT Volume exist

# dataSetIDPath = "/home/hxie1/data/BES_3K/GTs/testID.csv"
rawPath = "/home/hxie1/data/BES_3K/raw"

import glob
import fnmatch
import sys


def main():
    # get csv file name
    dataSetIDPath = sys.argv[1]

    # get all ID List
    with open(dataSetIDPath,'r') as f:
        IDList = f.readlines()
    IDList = [item[0:-1] for item in IDList] # erase '\n'
    print(f"total {len(IDList)} IDs in {dataSetIDPath}")

    # get all volume List
    volumeODList = glob.glob(rawPath + f"/*_OD_*_Volume")
    print(f"total {len(volumeODList)} OD volumes in {rawPath}")

    N= 0
    nonexistIDList = []
    for ID in IDList:
        resultList = fnmatch.filter(volumeODList, "*/"+ID+"_OD_*_Volume")
        length = len(resultList)
        if 0== length:
            nonexistIDList.append(ID)
        elif length > 1:
            print(f"Mulitple ID files: {resultList}")
        else:
            N +=1
    print(f"find {N} corresponding volumes")

if __name__ == "__main__":
    main()
