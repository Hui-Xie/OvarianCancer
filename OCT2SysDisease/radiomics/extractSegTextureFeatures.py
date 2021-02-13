# Extract segmented OCT layer texture features.

srcVolumePath ="/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes/2081_OD_14279_Volume.npy"
segXmlPath ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml/2081_OD_14279_Volume_Sequence_Surfaces_Prediction.xml"
outputDir = "/home/hxie1/temp/extractRadiomics"

radiomicsCfgPath = "/home/hxie1/projects/DeepLearningSeg/OCT2SysDisease/radiomics/testConfig/OCTLayerTextureCfg.yaml"
indexBscan = 15


import numpy as np
# import os
from PIL import Image

import sys
sys.path.append("../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import  getSurfacesArray

import logging
import os

import radiomics
from radiomics import featureextractor, getFeatureClasses


def generateImage_Mask(volumePath, xmlPath, indexBscan, outputDir):
    volumeName, _ = os.path.splitext(os.path.basename(volumePath))
    sliceName = volumeName + f"_s{indexBscan}"

    imagePath = os.path.join(outputDir, sliceName + f"_texture.tif")  # It must use tiff format for float format
    maskPath = os.path.join(outputDir, sliceName + f"_mask.tif")

    volume = np.load(volumePath)  # 31x496x512
    volumeSeg  = getSurfacesArray(xmlPath).astype(np.uint32)  # 31x10x512
    slice = volume[indexBscan,]  # 496x512
    H,W = slice.shape
    sliceSeg = volumeSeg[indexBscan,]  # 10x512
    N,W = sliceSeg.shape

    #generate retina layer mask
    mask = np.zeros(slice.shape, dtype=np.uint32)  # size: HxW
    for c in range(W):
        mask[sliceSeg[0,c]:sliceSeg[N-1,c],c] = 1


    # use PIL to save image
    Image.fromarray(slice).save(imagePath)
    Image.fromarray(mask).save(maskPath)

    return imagePath, maskPath

def generateRadiomics(imagePath, maskPath, radiomicsCfgPath):
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
    for featureName in featureVector.keys():
        if "diagnostics_" == featureName[0:12]:
            print(f"{featureName}:{featureVector[featureName]}")
    print("========================================")
    print(f"\nPrint original features:")
    for featureName in featureVector.keys():
        if "original_" == featureName[0:9]:
            print(f"{featureName}:{featureVector[featureName]}")
    print("========================================")


def main():
    imagePath, maskPath = generateImage_Mask(srcVolumePath, segXmlPath, indexBscan, outputDir)
    generateRadiomics(imagePath,maskPath, radiomicsCfgPath)


if __name__ == "__main__":
    main()
