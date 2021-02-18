# extract 3D spacing of raw image and output in to yaml file
segXmlDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml"
rawVolumeDir ="/home/hxie1/data/BES_3K/raw"

outputDir = "/home/hxie1/data/BES_3K/GTs/spacing"
# output yaml format:
# volumeName: [W,H,Z] # spacing in mm/pixel

import glob
import os
import sys
import datetime

output2File = True
hSpacing = 0.003870 # mm/pixel

def main():
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(outputDir, f"output_extractSpacing_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    print(f"=============== Extract the spacing of raw volumes ================")

    patientSegsList = glob.glob(segXmlDir + f"/*_Volume_Sequence_Surfaces_Prediction.xml")
    print(f"total {len(patientSegsList)} xml files.")
    for xmlPath in patientSegsList:
        volumeName = os.path.splitext(os.path.basename(xmlPath))[0]  # 370_OD_458_Volume_Sequence_Surfaces_Prediction
        volumeName = volumeName[0:volumeName.find("_Sequence_Surfaces_Prediction")]  # 370_OD_458_Volume
        aVolumeDir = os.path.join(rawVolumeDir, volumeName)

        infoPathList = glob.glob(aVolumeDir + f"/*_Info.txt")
        if len(infoPathList) != 0:
            print(f"at {aVolumeDir}, program found {len(infoPathList)} information files:\n {infoPathList}")
            continue

        infoPath = infoPathList[0]
        with open(infoPath) as f:
            lines = f.readlines()
            print("===debug===")




    if output2File:
        logOutput.close()
        sys.stdout = original_stdout

    print(f"===== End of extracting spacing of raw volumes ==============")


if __name__ == "__main__":
    main()