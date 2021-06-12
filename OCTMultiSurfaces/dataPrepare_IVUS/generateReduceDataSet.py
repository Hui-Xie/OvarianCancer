'''
Cut IVUS data into 10% for training set only.
1IVUS data set(100%):
training: 100 images.
validaiton: 9 images.
test:     326 images.

IVUS data set(10%):
training: 10 images.
validaiton: 9 images.
test:     326 images.

'''
H = 192
W = 360
S = 2   # number of surfaces

trainSrcDir = "/home/hxie1/data/IVUS/polarNumpy/training"

# 10%
outputDir10Percent = "/home/hxie1/data/IVUS/polarNumpy_10percent/training"


import os
import random
import numpy as np
import json

trainCutDict_10={"srcDir": trainSrcDir,
              "totalN": 100,
              "N": 10,
              "outputDir": outputDir10Percent }

def cutDataset(cutDict):
    srcDir= cutDict["srcDir"]
    IDPath = os.path.join(srcDir, "patientID.json")
    with open(IDPath) as f:
        totalIDs = json.load(f)
    assert (cutDict["totalN"] == len(totalIDs))

    # get cut Sample
    if cutDict["totalN"] == cutDict["N"]:
        cutSamples = list(range(0, cutDict["totalN"]))
    else:
        cutSamples = random.sample(list(range(0, cutDict["totalN"])), cutDict["N"])
        cutSamples.sort()

    assert (cutDict["N"] == len(cutSamples))
    N = cutDict["N"]

    patientIDDict = dict()

    originalImagesPath = os.path.join(srcDir,"images.npy")
    originalSurfacesPath = os.path.join(srcDir, "surfaces.npy")

    images = np.load(originalImagesPath).astype(float)[cutSamples,:,:]
    surfaces = np.load(originalSurfacesPath).astype(float)[cutSamples,:,:]

    i =0
    for x in cutSamples:
        patientIDDict[str(i)] = totalIDs[str(x)]
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
    print(f"validation and test set keeps same with 100% dataset.")
    print("===========END of Cut training data==========")


if __name__ == "__main__":
    main()

