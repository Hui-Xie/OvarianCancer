# generate K radiomics

textureDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/texture3D_nrrd_physicalSpace/texture"
maskDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/texture3D_nrrd_physicalSpace/mask_s3tos8"
outputRadiomicsDir="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/volume3D_s3tos8_100radiomics_physicalSpace"

radiomicsCfgPath = "/home/hxie1/projects/DeepLearningSeg/OCT2SysDisease/radiomics/testConfig/OCTLayerTextureCfg_100Radiomics_3D_physicalSpace.yaml"
K = 100   # the number of extracted features.
# surface index starting from zero.
sStart = 3
sEnd = 8

import glob
import numpy as np
import random

import os
from radiomics import featureextractor

def main():
    # Get the PyRadiomics logger (default log-level = INFO
    # logger = radiomics.logger
    # logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

    # Write out all log entries to a file
    # handler = logging.FileHandler(filename=os.path.join(outputRadiomicsDir, 'generateLog_radiomics.txt'), mode='w')
    # formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    textureList = glob.glob(textureDir + f"/*_Volume_texture.nrrd")
    random.shuffle(textureList)  # for multiple processes speed up.

    print(f"total {len(textureList)} texture files.")  # 6499 files.
    for texturePath in textureList:
        volumeName = os.path.basename(texturePath)  # 31118_OS_5069_Volume_texture.nrrd
        maskName = volumeName.replace("_texture.nrrd", f"_s{sStart}tos{sEnd}_mask.nrrd") # 31118_OS_5069_Volume_s3tos8_mask.nrrd
        maskPath = os.path.join(maskDir, maskName)

        radiomicsArrayName =volumeName.replace("_texture.nrrd", f"_{K}radiomics.npy")
        radiomicsArrayPath = os.path.join(outputRadiomicsDir, radiomicsArrayName)

        if os.path.isfile(radiomicsArrayPath):  # generating one file needs 30 seconds.
            continue

        # Initialize feature extractor using the settings file
        extractor = featureextractor.RadiomicsFeatureExtractor(radiomicsCfgPath)
        featureVector = extractor.execute(texturePath, maskPath, label=1)

        nFeatures = 0
        sortedFeatureKeys = sorted(featureVector.keys())  # make sure the output value in dictionary order.
        radiomicsArray = np.zeros((1, K), dtype=float)
        for featureName in sortedFeatureKeys:
            if "original_" == featureName[0:9]:
                # print(f"{featureName}:{featureVector[featureName]}")
                radiomicsArray[0, nFeatures] = featureVector[featureName]
                nFeatures += 1
        assert nFeatures==K

        if os.path.isfile(radiomicsArrayPath):  # generating one file needs 30 seconds.
            continue
        np.save(radiomicsArrayPath, radiomicsArray)
    print(f"===== End of generating {K} radiomics features ==============")


if __name__ == "__main__":
    main()