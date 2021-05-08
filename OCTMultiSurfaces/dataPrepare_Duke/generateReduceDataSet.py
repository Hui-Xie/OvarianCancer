'''
Cut Duke_AMD data into 10% and 20%.
1 Duke AMD data statistics:  51 Bscan / volume.
  training set: 266 volumes x 51 slices per volume, where 187 AMD + 79 control = 266 volumes;
  validation set: 59 volumes x 51 slice per volume, where 41 AMD + 18 control  =  59 volumes;
  test set:  59 volumes x 51 slice per volume, where 41 AMD + 18 control  =  59 volumes;
2 cut into 10%:
  training: 19 AMD + 8 control. (19+8)/266 = 10.1%.  Size: 26*51*496*512*4 = 1.35 GB
  validation: 4 AMD + 2 control. (4+2)/59  = 10.1%   size: 6*51*496*512*4  = 0.3 GB
  test set:   41 AMD + 18 control.                   size: 59*51*496*512*4 = 3.1 GB
3 cut into 20%:
  training:  38 AMD + 16 control. (38+16)/266 = 20.3%.  Size: 54*51*496*512*4 = 2.8 GB
  validation: 8 AMD + 4 control. (8+4)/59  =   20.3%   size: 12*51*496*512*4  = 0.6 GB
  test set:   41 AMD + 18 control.                     size: 59*51*496*512*4 = 3.1 GB
4 test set is same for full dataset.
5 output format:
  images.npy  patientID.json  surfaces.npy in test / validation /training directory.
  where: json file: ID->imageFileFullPath

'''
H = 496
W = 512
B = 51  # number of Bscans per volume
S = 3   # number of surfaces

trainSrcDir = "/home/hxie1/data/OCT_Duke/numpy/training"
validationSrcDir = "/home/hxie1/data/OCT_Duke/numpy/validation"
testSrcDir = "/home/hxie1/data/OCT_Duke/numpy/test"

# 10%
outputDir10Percent = "/home/hxie1/data/OCT_Duke/numpy_10Percent"
outputDir20Percent = "/home/hxie1/data/OCT_Duke/numpy_20Percent"

import os
import random
import numpy as np
import json

trainCutDict_10={"srcDir": trainSrcDir,
              "totalN": 266, "totalAMD":187, "totalControl": 79,
              "N": 27,        "AMD":19,      "Control":8,
              "outputDir": os.path.join(outputDir10Percent, "training") }

trainCutDict_20={"srcDir": trainSrcDir,
              "totalN": 266, "totalAMD":187, "totalControl": 79,
              "N": 54,        "AMD":38,      "Control":16,
              "outputDir": os.path.join(outputDir20Percent, "training") }

validationCutDict_10={"srcDir": validationSrcDir,
              "totalN": 59, "totalAMD":41, "totalControl": 18,
              "N": 6,        "AMD":4,      "Control":2,
              "outputDir": os.path.join(outputDir10Percent, "validation") }

validationCutDict_20={"srcDir": validationSrcDir,
              "totalN": 59, "totalAMD":41, "totalControl": 18,
              "N": 12,        "AMD":8,      "Control":4,
              "outputDir": os.path.join(outputDir20Percent, "validation") }

testCutDict ={"srcDir": testSrcDir,
              "totalN": 59, "totalAMD":41, "totalControl": 18,
              "N": 59,        "AMD":41,      "Control":18,
              "outputDir": os.path.join(outputDir10Percent, "test") }  # 10% and 20% has same test set.


def cutDataset(cutDict):
    srcDir= cutDict["srcDir"]
    patientIDListPath = os.path.join(srcDir, "patientList.txt")
    with open(patientIDListPath, 'r') as f:
        totalIDList = f.readlines()
    totalIDList = [item[0:-1] for item in totalIDList]
    assert (cutDict["totalN"] == len(totalIDList))

    # get cut Sample
    if cutDict["totalN"] == cutDict["N"]:
        cutSamples = list(range(0, cutDict["totalN"]))
    else:
        amdSamples = random.sample(list(range(0, cutDict["totalAMD"])), cutDict["AMD"])
        controlSamples = random.sample(list(range(cutDict["totalAMD"], cutDict["totalN"])), cutDict["Control"])
        cutSamples = amdSamples+ controlSamples

    assert (cutDict["N"] == len(cutSamples))
    N = cutDict["N"]

    images = np.empty((N*B, H,W),dtype=np.float32)
    surfaces = np.empty((N*B, S,W),dtype=np.float32)
    patientIDDict = dict()

    i =0
    for vIndex in cutSamples:
        volumePath = totalIDList[vIndex]
        surfacePath = volumePath[0:volumePath.find("_images.npy")] + "_surfaces.npy"
        patientID = volumePath[volumePath.rfind("/") : volumePath.find(".npy")]

        images[i*B:(i+1)*B,] = np.load(volumePath).astype(np.float32)
        surfaces[i*B:(i+1)*B,] = np.load(surfacePath).astype(np.float32)
        for s in range(i*B, (i+1)*B):
            patientIDDict[str(s)] = patientID+f"_s{s-i*B:02d}.npy"
        i += 1

    # save
    if not os.path.exists(cutDict["outputDir"]):
        os.makedirs(cutDict["outputDir"])

    np.save(os.path.join(cutDict["outputDir"], "images.npy"), images)
    np.save(os.path.join(cutDict["outputDir"], "surfaces.npy"), surfaces)
    with open(os.path.join(cutDict["outputDir"], "patientID.json"), 'w') as fp:
        json.dump(patientIDDict, fp)


def main():
    cutDataset(trainCutDict_10)
    cutDataset(trainCutDict_20)
    cutDataset(validationCutDict_10)
    cutDataset(validationCutDict_20)
    cutDataset(testCutDict)
    print(f"for the test directory, you need to copy from 10% to 20%")


if __name__ == "__main__":
    main()
