# extract good Bscans from output directory.

import yaml
import glob
import os

srcPath = "/home/hxie1/data/OCT_Tongren/goodBscanPrediction_20200512/10Surfaces_31BscansForEachPatient"
outputPath ="/home/hxie1/data/OCT_Tongren/goodBscanPrediction_20200512/10Surfaces_JustGoodBscansForEachPatient"
yamlFilePath = "/home/hxie1/Projects/DeepLearningSeg/OCTMultiSurfaces/testConfig/expTongren_10Surfaces_allGoodBscans_20200512/allGoodBscans.yaml"

# read good bscans config
# read yaml file
with open(yamlFilePath) as file:
    goodBscans = yaml.load(file, Loader=yaml.FullLoader)['goodBscans']

volumesList = glob.glob(srcPath + f"/*_Volume")
for volumePath in volumesList:
    volumeName = os.path.basename(volumePath)
    patientID = int(volumeName[0:volumeName.find("_OD_")])
    lowB = goodBscans[patientID][0] - 1;
    highB = goodBscans[patientID][1];

    outputVolumePath = outputPath +"/" + volumeName
    if not os.path.exists(outputVolumePath):
        os.makedirs(outputVolumePath)  # recursive dir creation
    imagesList = glob.glob(volumePath +"/*.png")
    imagesList.sort()

    for b in range(lowB, highB):
        command = 'cp '+ imagesList[b] +" " + outputVolumePath+"/"+os.path.basename(imagesList[b])
        os.popen(command)
