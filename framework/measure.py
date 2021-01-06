
import math
import csv
import torch

import numpy as np

# for integer classification: residual tumor size, and chemo response
def computeClassificationAccuracy(gtDict, predictDict, key):
    '''
    predict.keys may be less than gtDice.keys.
    :param gtDict:
    :param predictDict:
    :param key:
    :return:
    '''
    predictKeys = list(predictDict.keys())

    countSame = 0
    countDiff = 0
    countIgnore = 0

    for ID in predictKeys:
        if gtDict[ID][key] == -100:
            countIgnore +=1
        elif gtDict[ID][key] == predictDict[ID][key]:
            countSame +=1
        else:
            countDiff +=1
    N = countSame + countDiff +countIgnore
    acc = countSame*1.0/(countSame + countDiff)

    return acc

def computeClassificationAccuracyWithLogit(gt, predictLogits):
    '''
    tensor version.
    :param gt: in torch.tensor in 1D
    :param predictLogits: in torch.tensor in 1 D
    :return:
    '''
    assert gt.shape == predictLogits.shape

    predict = (torch.sigmoid(predictLogits) +0.5).int()
    gt = gt.int()

    N = predict.numel()
    countDiff = torch.count_nonzero(gt-predict).item()

    acc = (N - countDiff)*1.0/N

    return acc

# for linear regression prediction: age, survival months
def computeSqrtMSE(gtDict, predictDict, key):
    predictKeys = list(predictDict.keys())
    mse = 0.0
    nCount = 0

    for ID in predictKeys:
        if gtDict[ID][key] == -100:
            continue
        else:
            mse += (gtDict[ID][key] - predictDict[ID][key])**2
            nCount +=1
    mse = mse*1.0/nCount
    sqrtMse = math.sqrt(mse)
    return sqrtMse

def outputPredictProbDict2Csv(predictProbDict, csvPath):
    '''
        csv data example:
        MRN,Prob1,GT,
        6501461,0.3,0.7,0,

    '''
    with open(csvPath, "w") as file:
        file.write("ID,Prob1,GT,\n")
        for key in predictProbDict:
            a = predictProbDict[key]
            file.write(f"{key},{a['Prob1']},{a['GT']},\n")


def readProbDict(csvPath):
    '''
       csv data example:
        MRN,Prob0, Prob1, GT,
        6501461,0.3,0.7,0,

    '''
    probDict = {}
    with open(csvPath, newline='') as csvfile:
        csvList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
        csvList = csvList[1:]  # erase table head
        for row in csvList:
            MRN = '0' + row[0] if 7 == len(row[0]) else row[0]
            probDict[MRN] = {}
            probDict[MRN]['Prob1'] = float(row[1]) if 0 != len(row[1]) else -100
            probDict[MRN]['GT'] = int(row[2]) if 0 != len(row[2]) else -100
    return probDict

def computeThresholdAccTPR_TNRSumFromProbDict(probDict):
    '''
    probDict format: key, Prob1, GT  without table head.
    
    :param probDict: 
    :return: (Threhold, ACC, TPR, TNR, Sum) at max(Acc+TPR+TNR).
    '''

    epsilon = 1e-8
    keys = list(probDict.keys())
    nTotal = len(keys)

    tdList = np.arange(0.001, 0.999, 0.002)
    N = len(tdList)
    ACCList = [0]*N
    TPRList = [0]*N
    TNRList = [0]*N
    SumList = [0]*N
    for i,td in enumerate(tdList):  # td: threshold
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        nIgnore = 0

        for MRN in keys:
            if probDict[MRN]['GT'] == -100:
                nIgnore += 1
                continue
            if probDict[MRN]['Prob1'] < td:
                if probDict[MRN]['GT'] == 0:
                    TN += 1
                else:
                    FN += 1
            else:
                if probDict[MRN]['GT'] == 0:
                    FP += 1
                else:
                    TP += 1

        assert nTotal == TP + TN + FP + FN + nIgnore
        ACCList[i] = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        TPRList[i] = TP * 1.0 / (TP + FN + epsilon)  # sensitivity
        TNRList[i] = TN * 1.0 / (TN + FP + epsilon)  # specificity
        SumList[i] = ACCList[i] + TPRList[i] + TNRList[i]

    maxIndex = np.argmax(SumList)
    return {"threshold": tdList[maxIndex], "ACC": ACCList[maxIndex], "TPR":TPRList[maxIndex], "TNR": TNRList[maxIndex], "Sum":SumList[maxIndex]}


def search_Threshold_Acc_TPR_TNR_Sum_WithLogits(gt,predictLogits):
    '''
    tensor version
    :param gt: in torch.tensor in 1D
    :param predictLogits: in torch.tensor in 1 D
    :return: (Threhold, ACC, TPR, TNR, Sum) at max(Acc+TPR+TNR).
    '''

    epsilon = 1e-8
    assert gt.shape == predictLogits.shape

    predictProb = torch.sigmoid(predictLogits)
    gt = gt.int()
    nTotal = predictProb.numel()

    tdList = np.arange(0.001, 0.999, 0.002) # td: threshold
    N = len(tdList)
    ACCList = [0] * N
    TPRList = [0] * N
    TNRList = [0] * N
    SumList = [0] * N
    for i, td in enumerate(tdList):  # td: threshold
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        nIgnore = 0

        for n in range(nTotal):
            if predictProb[n] < td:
                if gt[n] == 0:
                    TN += 1
                else:
                    FN += 1
            else:
                if gt[n] == 0:
                    FP += 1
                else:
                    TP += 1

        assert nTotal == TP + TN + FP + FN + nIgnore
        ACCList[i] = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        TPRList[i] = TP * 1.0 / (TP + FN + epsilon)  # sensitivity
        TNRList[i] = TN * 1.0 / (TN + FP + epsilon)  # specificity
        SumList[i] = ACCList[i] + TPRList[i] + TNRList[i]

    maxIndex = np.argmax(SumList)
    return {"threshold": tdList[maxIndex], "ACC": ACCList[maxIndex], "TPR": TPRList[maxIndex], "TNR": TNRList[maxIndex],
            "Sum": SumList[maxIndex]}


def compute_Acc_TPR_TNR_Sum_WithLogits(gt,predictLogits, threshold):
    '''
    tensor version
    :param gt: in torch.tensor in 1D
    :param predictLogits: in torch.tensor in 1 D
    :return: (ACC, TPR, TNR, Sum)
    '''

    epsilon = 1e-8
    assert gt.shape == predictLogits.shape

    predictProb = torch.sigmoid(predictLogits)
    gt = gt.int()
    nTotal = predictProb.numel()

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    nIgnore = 0
    for n in range(nTotal):
        if predictProb[n] < threshold:
            if gt[n] == 0:
                TN += 1
            else:
                FN += 1
        else:
            if gt[n] == 0:
                FP += 1
            else:
                TP += 1

    assert nTotal == TP + TN + FP + FN + nIgnore
    ACC = (TP + TN) * 1.0 / (TP + TN + FP + FN)
    TPR = TP * 1.0 / (TP + FN + epsilon)  # sensitivity
    TNR = TN * 1.0 / (TN + FP + epsilon)  # specificity
    Sum = ACC + TPR + TNR

    return {"ACC": ACC, "TPR": TPR, "TNR": TNR, "Sum": Sum}
