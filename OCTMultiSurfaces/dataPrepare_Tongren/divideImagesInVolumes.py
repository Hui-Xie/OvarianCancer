
# divide all image into volumes

import glob
import os

srcPath = "/home/hxie1/data/OCT_Tongren/goodBscanPrediction_20200512/10Surfaces_31BscansForEachPatient"
outputPath = srcPath

imagesList = glob.glob(srcPath + f"/*_Raw_GT_Predict.png")
imagesList.sort()

for imagePath in imagesList:
    imageBasename = os.path.basename(imagePath)
    volumeName =imageBasename[0: imageBasename.find("_2011")]
    outputVolumePath = outputPath + "/" + volumeName
    if not os.path.exists(outputVolumePath):
        os.makedirs(outputVolumePath)  # recursive dir creation
    command = 'mv ' + imagePath + " " + outputVolumePath
    os.popen(command)

