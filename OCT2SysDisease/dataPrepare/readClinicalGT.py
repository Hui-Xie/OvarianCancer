
# read BES clinical data

gtPath= "/home/hxie1/data/BES_3K/GTs/BESClinicalGT.csv"

import sys
sys.path.append("..")
from network.OCT2SysDiseaseTools import readBESClinicalCsv

def main():
    gtDict = readBESClinicalCsv(gtPath)
    print(f"gtDict length = {len(gtDict)}")
    print(f"gtDict[13] = {gtDict[13]}")

if __name__ == "__main__":
    main()