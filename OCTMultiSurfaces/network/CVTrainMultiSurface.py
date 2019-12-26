
import sys
import yaml
import os


import torch
sys.path.append(".")
from OCTDataSet import OCTDataSet



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
    trainLabelsPath  = os.path.join(dataDir,"training", f"surfaces_CV{k:d}.npy")
    trainIDPath     = os.path.join(dataDir,"training", f"patientID_CV{k:d}.json")

    validationImagesPath = os.path.join(dataDir,"validation", f"images_CV{k:d}.npy")
    validationLabelsPath = os.path.join(dataDir,"validation", f"surfaces_CV{k:d}.npy")
    validationIDPath    = os.path.join(dataDir,"validation", f"patientID_CV{k:d}.json")

    trainDataSet = OCTDataSet(trainImagesPath, trainLabelsPath, trainIDPath, transform=None, device=device, sigma=sigma)
    validationDataSet = OCTDataSet(validationImagesPath, validationLabelsPath, validationIDPath, transform=None, device=device, sigma=sigma)

    x = trainDataSet.__getitem__(30)

    print("============ End of Cross valiation training for OCT Multisurface Network ===========")



if __name__ == "__main__":
    main()