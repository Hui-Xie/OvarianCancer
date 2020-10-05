
import math
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
