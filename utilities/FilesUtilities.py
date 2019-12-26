# define some files utilities method

import os

def getFilesList(filesDir, suffix):
    originalCwd = os.getcwd()
    os.chdir(filesDir)
    filesList = [os.path.abspath(x) for x in os.listdir(filesDir) if suffix in x]
    os.chdir(originalCwd)
    return filesList

def saveInputFilesList(filesList, filename):
    with open( filename, "w") as f:
        for file in filesList:
            f.write(file + "\n")

def loadInputFilesList(filename):
    filesList = []
    with open( filename, "r") as f:
        for line in f:
            filesList.append(line.strip())
    return filesList

def getStemName(path, removedSuffix=None):
    baseName = os.path.basename(path)
    if removedSuffix is None:
        base = baseName
    else:
        base = baseName[0: baseName.find(removedSuffix)]
    return base

def getFinalLine(filename):
    with open(filename, 'rb') as f:
        for line in f:
            pass
    return line.decode("utf-8")

def getListFromLine(line):
    line = line.replace('\t\t', '\t')
    row = line.split('\t')
    return row
