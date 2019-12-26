
import sys
import yaml
import os

import torch


def printUsage(argv):
    print("============ Cross Validation Train OCT MultiSurface Network =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")

def main():

    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # parse config file
    configFile = sys.argv[1]
    with open(configFile) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    dataDir = cfg["dataDir"]
    K = cfg["K_Folds"]
    k = cfg["fold_k"]
    sigma = cfg["sigma"]  # for gausssian ground truth
    device = eval(cfg["device"])  # convert string to class object.
    batchSize = cfg["batchSize"]

    trainImagesPath = os.path.join(dataDir,"training", f"images_CV{k:d}.npy")
    trainLabelPath  = os.path.join(dataDir,"training", f"surfaces_CV{k:d}.npy")
    validationImagePath = os.path.join(dataDir,"validation", f"images_CV{k:d}.npy")
    validationLabelPath = os.path.join(dataDir,"validation", f"surfaces_CV{k:d}.npy")

    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()