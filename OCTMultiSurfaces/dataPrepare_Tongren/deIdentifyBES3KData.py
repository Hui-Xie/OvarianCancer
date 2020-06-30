# de-Identify BES 3K data

import glob as glob
import os

volumesDir = "/home/hxie1/data/BES_3K/raw"
#volumesDir = "/home/hxie1/temp"
volumesList = glob.glob(volumesDir + f"/*_Volume")
oldCurDir = os.getcwd()
for volumeDir in volumesList:
    os.chdir(volumeDir)
    filesList = glob.glob(volumeDir + f"/*.*")
    for filePath in filesList:
        fileName = os.path.basename(filePath)
        newFileName = fileName[fileName.find('_')+1:]
        os.system(f"mv {fileName} {newFileName}")
    os.chdir(volumesDir)
    volumeDir = os.path.basename(volumeDir)
    newVolumeDir = volumeDir[volumeDir.find('_')+1:]
    os.system(f"mv {volumeDir} {newVolumeDir}")

os.chdir(oldCurDir)
print("===Finished the deIdentity of BES 3K data=======")
