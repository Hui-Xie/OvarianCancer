# Grid search probability threshold

csvPath = "/home/hxie1/data/OvarianCancerCT/survivalPredict/log/ResponseNet/expOV_20200925_O_NoAugmentation/testResult/trainSetPredict_20200925_163534.csv"

import numpy as np

import sys
sys.path.append('..')
from network.OVTools import readProbDict

probDict = readProbDict(csvPath)

epsilon =1e-8

print("ProbThreshold,\tACC,\tTPR,\tTNR,")
for td in np.arange(0.05, 0.95, 0.05):  # td: threshlod
    keys = list(probDict.keys())
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

    ACC = (TP+TN)*1.0/(TP+TN+FP+FN)
    TPR = TP*1.0/(TP+FN+epsilon)  # sensitivity
    TNR = TN*1.0/(TN+FP+epsilon)  # specificity
    print(f"{td},\t{ACC},\t{TPR},\t{TNR},")
