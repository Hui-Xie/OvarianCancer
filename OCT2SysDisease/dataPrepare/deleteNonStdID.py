
# these ID has image size [496,384], instead of [496, 512]
notStdImageID = ['120035', '123', '1536', '1703', '33019', '34174', '34540', '34611', '378', '4479', '5152', '6253']

import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        print("Usage: command IDfile.csv")
        return -1

    IDPath = sys.argv[1]

    # read ID file
    with open(IDPath, 'r') as idFile:
        IDList = idFile.readlines()
    IDList = [item[0:-1] for item in IDList]  # erase '\n'
    oldLength = len(IDList)

    # delete some IDs
    for errID in notStdImageID:
        IDList.remove(errID)
    newLength  = len(IDList)

    print(f"deleted {newLength - oldLength} IDs in {IDPath}")
    # write ID file
    filename, ext = os.path.splitext(IDPath)
    outputFilename = filename +"_delErrWID" + ext
    with open(outputFilename, "w") as file:
        for id in IDList:
            file.write(f"{id}\n")

if __name__ == "__main__":
    main()
