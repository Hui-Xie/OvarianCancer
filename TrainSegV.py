import sys
from DataMgr import DataMgr


def main():
    if len(sys.argv) != 3:
        print("Error: input parameters error.")
        return -1
    else:
        dataMgr = DataMgr(sys.argv[1], sys.argv[2])
        filesList = dataMgr.getFilesList(dataMgr.m_imagesDir, "_CT.nrrd")
        print(filesList)
        print(len(filesList))

if __name__ == "__main__":
    main()
