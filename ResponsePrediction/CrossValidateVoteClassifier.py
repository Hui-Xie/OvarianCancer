

import sys
import yaml
import torch

def printUsage(argv):
    print("============ Cross Validation Vote Classifier =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_path")


def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    configFile = sys.argv[1]
    with open(configFile) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    latentDir = cfg["latentDir"]
    suffix = cfg["suffix"]
    patientResponsePath = cfg["patientResponsePath"]
    K_folds = cfg["K_folds"]
    fold_k = cfg["fold_k"]
    netPath = cfg["netPath"]
    rawF = cfg["rawF"]
    F = cfg["F"]
    device = eval(cfg["device"])  # convert string to class object.
    featureIndices = cfg["featureIndices"]

    print("================End of Cross Validation==============")

if __name__ == "__main__":
    main()
