# crop raw images with width 512, and save it in outputDir.

gtDir = "/home/hxie1/data/OCT_Tongren/refinedGT_20200204"  # corrected result by Tongren doctors, 47 files
rawImagesDir = "/home/hxie1/data/OCT_Tongren/control"
outputDir = "/home/hxie1/data/OCT_Tongren/control_W512Crop"

import glob as glob
import os
from imageio import imread, imwrite



#get gt patients list
patientSegsList = glob.glob(gtDir + f"/*_Volume_Sequence_Surfaces_Iowa.xml")
patientsList = []
for segFile in patientSegsList:
    patientSurfaceName = os.path.splitext(os.path.basename(segFile))[0]  # e.g. 1062_OD_9512_Volume_Sequence_Surfaces_Iowa
    patientVolumeName = patientSurfaceName[0:patientSurfaceName.find("_Sequence_Surfaces_Iowa")]  # 1062_OD_9512_Volume
    patientsList.append(rawImagesDir + f"/{patientVolumeName}")

#read patient image, crop, save.
for volume in patientsList:
    imagesList = glob.glob(volume + f"/*_OCT[0-3][0-9].jpg")
    if  31 !=len(imagesList):
        print(f"{volume} does not has 31 jpg files")
        break
    imagesList.sort()
    outputVolumeDir = os.path.join(outputDir, os.path.basename(volume))
    if not os.path.exists(outputVolumeDir):
        os.makedirs(outputVolumeDir)  # recursive dir creation
    for imagePath in imagesList:
        if "5363_OD_25453" in  volume:
            image = imread(imagePath)[:, 103:615]
        else:
            image = imread(imagePath)[:, 128:640]
        imwrite(os.path.join(outputVolumeDir,os.path.basename(imagePath)), image)

print("====End of Cropping Images with Width 512==============")

