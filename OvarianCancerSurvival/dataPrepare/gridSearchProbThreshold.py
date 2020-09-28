# Grid search probability threshold

import numpy as np

import sys
sys.path.append('..')
from network.OVTools import readProbDict

def main():
    # get csv file name
    csvPath = sys.argv[1]

    probDict = readProbDict(csvPath)

    epsilon =1e-8

    print("ProbThreshold,\tACC,\tTPR,\tTNR,\tSum(ACC_TPR_TNR),")
    for td in np.arange(0.001, 0.999, 0.002):  # td: threshlod
        keys = list(probDict.keys())
        nTotal = len(keys)
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for MRN in keys:
            if probDict[MRN]['Prob1'] < td:
                if  probDict[MRN]['GT'] == 0:
                    TN +=1
                else:
                    FN +=1
            else:
                if  probDict[MRN]['GT'] == 0:
                    FP +=1
                else:
                    TP +=1

        assert nTotal == TP+TN+FP+FN
        ACC = (TP+TN)*1.0/(TP+TN+FP+FN)
        TPR = TP*1.0/(TP+FN+epsilon)  # sensitivity
        TNR = TN*1.0/(TN+FP+epsilon)  # specificity
        print(f"{td:.3f},\t{ACC:.2f},\t{TPR:.2f},\t{TNR:.2f},\t{ACC+TPR+TNR:.3f},")


if __name__ == "__main__":
    main()
