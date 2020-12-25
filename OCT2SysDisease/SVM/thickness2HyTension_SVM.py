# Use SVM method to predict hypertension from thickness map

from sklearn import svm
import sys
sys.path.append("../..")
from framework.ConfigReader import ConfigReader

def printUsage(argv):
    print("============ OCT Thickness to Hypertension Using SVM =============")
    print("Usage:")
    print(argv[0], " yaml_Config_file_full_path")

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1

    # parse config file
    configFile = sys.argv[1]
    hps = ConfigReader(configFile)
    print(f"Experiment: {hps.experimentName}")


    # load
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf = svm.SVC()
    clf.fit(X, y)

    prediction =clf.predict([[2., 2.]])
    print(f"prediction: {prediction}")

if __name__ == "__main__":
    main()