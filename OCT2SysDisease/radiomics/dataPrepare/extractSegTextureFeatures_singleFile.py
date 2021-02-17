# Extract segmented OCT layer texture features.

srcVolumePath ="/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes/6890_OD_19307_Volume.npy"
segXmlPath ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/xml/6890_OD_19307_Volume_Sequence_Surfaces_Prediction.xml"
outputDir = "/home/hxie1/temp/extractRadiomics"

radiomicsCfgPath = "/home/hxie1/projects/DeepLearningSeg/OCT2SysDisease/radiomics/testConfig/OCTLayerTextureCfg_95Radiomics_2D.yaml"
indexBscan = 15
K = 95   # the number of extracted features.


import numpy as np
from PIL import Image

import sys
sys.path.append("../../..")
from OCTMultiSurfaces.dataPrepare_Tongren.TongrenFileUtilities import  getSurfacesArray

import logging
import os
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor, getFeatureClasses


def generateImage_Mask(volumePath, xmlPath, indexBscan, outputDir):
    volumeName, _ = os.path.splitext(os.path.basename(volumePath))
    sliceName = volumeName + f"_s{indexBscan}"

    # use Tiff to save image and mask
    # pyradiomics can not correctly recognize mask in 2D tiff.
    # imagePath = os.path.join(outputDir, sliceName + f"_texture.tif")
    # maskPath = os.path.join(outputDir, sliceName + f"_mask.tif")

    # use nrrd to save image and mask
    imagePath = os.path.join(outputDir, sliceName + f"_texture.nrrd")
    maskPath = os.path.join(outputDir, sliceName + f"_mask.nrrd")

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
    # Image.fromarray(slice).save(imagePath)
    # Image.fromarray(mask).save(maskPath)

    # use sitk to save image
    slice = np.expand_dims(slice,axis=0)
    sitk.WriteImage(sitk.GetImageFromArray(slice), imagePath)
    mask = np.expand_dims(mask, axis=0)  # expand into 3D array
    sitk.WriteImage(sitk.GetImageFromArray(mask), maskPath)


    return imagePath, maskPath

def generateRadiomics(imagePath, maskPath, radiomicsCfgPath):
    sliceName,_ = os.path.splitext(os.path.basename(imagePath))  # 2081_OD_14279_Volume_s15_texture
    sliceName = sliceName[0:sliceName.rfind("_texture")]

    radiomicsArrayPath = os.path.join(outputDir, sliceName+f"_{K}radiomics.npy")

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
    extractor.settings["force2D"] = True
    extractor.settings["force2Ddimension"] = 0
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
    print("============================================")

    print(f"\nPrint original features:")
    nFeatures = 0
    sortedFeatureKeys = sorted(featureVector.keys())  # make sure the output value in dictionary order.
    radiomicsArray = np.zeros((1,K), dtype=np.float32)
    for featureName in sortedFeatureKeys:
        if "original_" == featureName[0:9]:
            print(f"{featureName}:{featureVector[featureName]}")
            radiomicsArray[0, nFeatures] = featureVector[featureName]
            nFeatures += 1
    np.save(radiomicsArrayPath,radiomicsArray)
    print("=============================================")
    print(f"===========Total {nFeatures} features=============")


def main():
    imagePath, maskPath = generateImage_Mask(srcVolumePath, segXmlPath, indexBscan, outputDir)
    print(f"imagePath = {imagePath}")
    print(f"maskPath  = {maskPath}")

    # use new bath dir
    #imagePath = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15TextureMask/texture/6890_OD_19307_Volume_s15_texture.nrrd"
    #maskPath = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15TextureMask/mask/6890_OD_19307_Volume_s15_mask.nrrd"

    # use temp dir:
    #imagePath = "/home/hxie1/temp/extractRadiomics/6890_OD_19307_Volume_s15_texture.nrrd"
    #maskPath = "/home/hxie1/temp/extractRadiomics/6890_OD_19307_Volume_s15_mask.nrrd"
    generateRadiomics(imagePath,maskPath, radiomicsCfgPath)
    print("=====End===")


if __name__ == "__main__":
    main()
