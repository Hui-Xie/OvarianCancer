# check whether the OCT Volume exist

# dataSetIDPath = "/home/hxie1/data/BES_3K/GTs/testID.csv"
rawPath = "/home/hxie1/data/BES_3K/raw"

import glob
import fnmatch
import sys
import os


def main():
    # get csv file name
    dataSetIDPath = sys.argv[1]

    # get all ID List
    with open(dataSetIDPath,'r') as f:
        IDList = f.readlines()
    IDList = [item[0:-1] for item in IDList] # erase '\n'
    rawIDLen = len(IDList)
    print(f"total {rawIDLen} IDs in {dataSetIDPath}")

    # get all volume List
    volumeODList = glob.glob(rawPath + f"/*_OD_*_Volume")
    print(f"total {len(volumeODList)} OD volumes in {rawPath}")

    N= 0
    nonexistIDList = []
    existIDList = []
    for ID in IDList:
        resultList = fnmatch.filter(volumeODList, "*/"+ID+"_OD_*_Volume")
        length = len(resultList)
        if 0== length:
            nonexistIDList.append(ID)
        elif length > 1:
            print(f"Mulitple ID files: {resultList}")
        else:
            existIDList.append(ID)
            N +=1
    print(f"find {N} corresponding volumes with ID")
    print(f"NonExistIDList: {nonexistIDList}")
    print(f"total {len(nonexistIDList)} IDs nonexist")

    # output updated existID list
    if N < rawIDLen:
        filepath, ext = os.path.splitext(dataSetIDPath)
        updatedIDPath = filepath + "_delNonExist"+ ext
        with open(updatedIDPath, "w") as file:
            for ID in existIDList:
                file.write(f"{ID}\n")
        print(f"output {updatedIDPath}")

if __name__ == "__main__":
    main()
