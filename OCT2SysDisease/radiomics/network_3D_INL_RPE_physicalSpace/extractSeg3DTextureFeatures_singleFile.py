# Extract segmented OCT layer texture features.


imagePath = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/texture3D_nrrd_physicalSpace/texture/31003_OD_932_Volume_texture.nrrd"
maskPath = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/texture3D_nrrd_physicalSpace/mask_s3tos8/31003_OD_932_Volume_s3tos8_mask.nrrd"
outputDir = "/home/hxie1/temp/extract3DRadiomics_physicalSpace"

radiomicsCfgPath = "/home/hxie1/projects/DeepLearningSeg/OCT2SysDisease/radiomics/testConfig/OCTLayerTextureCfg_100Radiomics_3D_physicalSpace.yaml"

K = 100 # need to change.

import numpy as np
from PIL import Image

import sys
sys.path.append("../../..")

import logging
import os
import SimpleITK as sitk
import datetime

import radiomics
from radiomics import featureextractor, getFeatureClasses

output2File = True

def generateRadiomics(imagePath, maskPath, radiomicsCfgPath):
    volumeName,_ = os.path.splitext(os.path.basename(imagePath))  # 297_OD_2031_Volume_texture
    volumeName = volumeName[0:volumeName.rfind("_texture")]

    radiomicsArrayPath = os.path.join(outputDir, volumeName+f"_3Dradiomics.npy")

    # Get the PyRadiomics logger (default log-level = INFO
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    # Write out all log entries to a file
    handler = logging.FileHandler(filename=os.path.join(outputDir, 'testLog_radiomics.txt'), mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Initialize feature extractor using the settings file
    extractor = featureextractor.RadiomicsFeatureExtractor(radiomicsCfgPath)
    featureClasses = getFeatureClasses()

    print("Active features:")
    for cls, features in extractor.enabledFeatures.items():
        if features is None or len(features) == 0:
            features = [f for f, deprecated in featureClasses[cls].getFeatureNames().items() if not deprecated]
        for f in features:
            print(f)
            print(getattr(featureClasses[cls], 'get%sFeatureValue' % f).__doc__)

    print("Calculating features")
    featureVector = extractor.execute(imagePath, maskPath, label=1)

    print(f"Print diagnostics features:")
    nDiagnostics = 0
    for featureName in featureVector.keys():
        if "diagnostics_" == featureName[0:12]:
            print(f"{featureName}:{featureVector[featureName]}")
            nDiagnostics += 1
    print(f"=============total {nDiagnostics} diagnostics features===============================")

    print(f"\nPrint original features:")
    nFeatures = 0
    sortedFeatureKeys = sorted(featureVector.keys())  # make sure the output value in dictionary order.
    # radiomicsArray = np.zeros((1,K), dtype=float)
    for featureName in sortedFeatureKeys:
        if "original_" == featureName[0:9]:
            print(f"{featureName}:{featureVector[featureName]}")
            # radiomicsArray[0, nFeatures] = featureVector[featureName]
            nFeatures += 1
    # np.save(radiomicsArrayPath,radiomicsArray)
    print(f"===========Total {nFeatures} original features=============")


def main():
    # prepare output file
    if output2File:
        curTime = datetime.datetime.now()
        timeStr = f"{curTime.year}{curTime.month:02d}{curTime.day:02d}_{curTime.hour:02d}{curTime.minute:02d}{curTime.second:02d}"

        outputPath = os.path.join(outputDir, f"output_generateRadiomicsPhysicalSpace_{timeStr}.txt")
        print(f"Log output is in {outputPath}")
        logOutput = open(outputPath, "w")
        original_stdout = sys.stdout
        sys.stdout = logOutput

    print(f"===============Generate radiomics feature from resample physical space image ================")

    generateRadiomics(imagePath,maskPath, radiomicsCfgPath)

    if output2File:
        logOutput.close()
        sys.stdout = original_stdout
    print(f"===============END=====================")

if __name__ == "__main__":
    main()
