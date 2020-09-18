

import math
import csv

# for integer classification: residual tumor size, and chemo response
def computeClassificationAccuracy(gtDict, predictDict, key):
    gtKeys = list(gtDict.keys())
    predictKeys = list(predictDict.keys())
    gtKeys.sort()
    predictKeys.sort()
    assert gtKeys == predictKeys

    countSame = 0
    countDiff = 0
    countIgnore = 0

    for MRN in gtKeys:
        if gtDict[MRN][key] == -100:
            countIgnore +=1
        elif gtDict[MRN][key] == predictDict[MRN][key]:
            countSame +=1
        else:
            countDiff +=1
    N = countSame + countDiff +countIgnore
    assert N == len(gtKeys)

    acc = countSame*1.0/N

    return acc

# for linear regression prediction: age, survival months
def computeSqrtMSE(gtDict, predictDict, key):
    gtKeys = list(gtDict.keys())
    predictKeys = list(predictDict.keys())
    gtKeys.sort()
    predictKeys.sort()
    assert gtKeys == predictKeys

    mse = 0.0
    nCount = 0

    for MRN in gtKeys:
        if gtDict[MRN][key] == -100:
            continue
        else:
            mse += (gtDict[MRN][key] - predictDict[MRN][key])**2
            nCount +=1
    mse = mse*1.0/nCount
    sqrtMse = math.sqrt(mse)
    return sqrtMse

def readGTDict(gtPath):
    '''
            csv data example:
            MRN,Age,ResidualTumor,Censor,TimeSurgeryDeath(d),ChemoResponse
            3818299,68,0,1,316,1
            5723607,52,0,1,334,0
            68145843,70,0,1,406,0
            4653841,64,0,1,459,0
            96776044,49,0,0,545,1

    '''
    gtDict = {}
    daysPerMonth = 30.4368
    with open(gtPath, newline='') as csvfile:
        csvList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
        csvList = csvList[1:]  # erase table head
        for row in csvList:
            MRN = '0' + row[0] if 7 == len(row[0]) else row[0]
            gtDict[MRN] = {}
            gtDict[MRN]['Age'] = int(row[1])
            gtDict[MRN]['ResidualTumor'] = int(row[2])
            # none data use -100 express
            gtDict[MRN]['Censor'] = int(row[3]) if 0 != len(row[3]) else -100
            gtDict[MRN]['SurvivalMonths'] = int(row[4]) / daysPerMonth if 0 != len(row[4]) else -100
            gtDict[MRN]['ChemoResponse'] = int(row[5]) if 0 != len(row[5]) else -100
    return gtDict


def outputPredictDict2Csv(predictDict, csvPath):
    '''
                csv data example:
                MRN,Age,ResidualTumor,Censor,TimeSurgeryDeath(d),ChemoResponse
                3818299,68,0,1,316,1
                5723607,52,0,1,334,0
                68145843,70,0,1,406,0
                4653841,64,0,1,459,0
                96776044,49,0,0,545,1

    '''
    with open(csvPath, "w") as file:
        file.write("MRN,Age,ResidualTumor,Censor,TimeSurgeryDeath(d),ChemoResponse,\n")
        daysPerMonth = 30.4368
        for key in predictDict:
            a = predictDict[key]
            file.write(f"{key},{a['Age']},{a['ResidualTumor']},{a['Censor']},{int(a['SurvivalMonths']*daysPerMonth)},{a['ChemoResponse']},\n")


