# generate 95 radiomics

textureDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15TextureMask/texture"
maskDir ="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15TextureMask/mask"
outputRadiomicsDir="/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/bscan15_95radiomics"

radiomicsCfgPath = "/home/hxie1/projects/DeepLearningSeg/OCT2SysDisease/radiomics/testConfig/OCTLayerTextureCfg_95Radiomics.yaml"
#indexBscan = 15
K = 95   # the number of extracted features.

import glob
import numpy as np

import os

import radiomics
from radiomics import featureextractor, getFeatureClasses

def main():
    textureList = glob.glob(textureDir + f"/*_Volume_s15_texture.tif")
    print(f"total {len(textureList)} texture files.")
    for texturePath in textureList:
        sliceName = os.path.basename(texturePath)  # 334027_OS_17817_Volume_s15_texture.tif
        maskName = sliceName.replace("_texture.tif", "_mask.tif") # 334027_OS_17817_Volume_s15_mask.tif
        maskPath = os.path.join(maskDir, maskName)

        radiomicsArrayName =sliceName.replace("_texture.tif", f"_{K}radiomics.npy")
        radiomicsArrayPath = os.path.join(outputRadiomicsDir, radiomicsArrayName)

        # Initialize feature extractor using the settings file
        extractor = featureextractor.RadiomicsFeatureExtractor(radiomicsCfgPath)
        featureVector = extractor.execute(texturePath, maskPath, label=1)

        nFeatures = 0
        sortedFeatureKeys = sorted(featureVector.keys())  # make sure the output value in dictionary order.
        radiomicsArray = np.zeros((1, K), dtype=np.float32)
        for featureName in sortedFeatureKeys:
            if "original_" == featureName[0:9]:
                # print(f"{featureName}:{featureVector[featureName]}")
                radiomicsArray[0, nFeatures] = featureVector[featureName]
                nFeatures += 1
        assert nFeatures==K
        np.save(radiomicsArrayPath, radiomicsArray)
    print(f"===== End of generating {K} radiomics ==============")


if __name__ == "__main__":
    main()