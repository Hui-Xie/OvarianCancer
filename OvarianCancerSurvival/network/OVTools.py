

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

    acc = countSame*1.0/(countSame + countDiff)

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

def readGTDict8Cols(gtPath):
    '''
        csv data example:
        MRN,Age,Optimal_surgery,Residual_tumor,Censor,Time_Surgery_Death,Response,Optimal_Outcome
        6501461,57,0,2,0,321,1,0
        5973259,52,0,2,1,648,1,0
        3864522,68,0,2,1,878,1,0
        5405166,71,0,2,1,892,1,0

        5612585,75,1,0,0,6,,1
        3930786,82,1,0,1,100,,1
        3848012,65,1,-1,1,265,,1
        85035058,76,1,0,0,,,1
        3020770,77,1,0,1,,,1

    '''
    gtDict = {}
    daysPerMonth = 30.4368
    with open(gtPath, newline='') as csvfile:
        csvList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
        csvList = csvList[1:]  # erase table head
        for row in csvList:
            MRN = '0' + row[0] if 7 == len(row[0]) else row[0]
            gtDict[MRN] = {}
            gtDict[MRN]['Age'] = int(row[1]) if 0 != len(row[1]) else -100
            gtDict[MRN]['OptimalSurgery'] = int(row[2]) if 0 != len(row[2]) else -100
            gtDict[MRN]['ResidualTumor'] = int(row[3]) if 0 != len(row[3]) else -100
            # none data use -100 express
            gtDict[MRN]['Censor'] = int(row[4]) if 0 != len(row[4]) else -100
            gtDict[MRN]['SurvivalMonths'] = int(row[5]) / daysPerMonth if 0 != len(row[5]) else -100
            gtDict[MRN]['ChemoResponse'] = int(row[6]) if 0 != len(row[6]) else -100
            gtDict[MRN]['OptimalResult'] = int(row[7]) if 0 != len(row[7]) else -100
    return gtDict



def outputPredictDict2Csv8Cols(predictDict, csvPath):
    '''
        csv data example:
        MRN,Age,Optimal_surgery,Residual_tumor,Censor,Time_Surgery_Death,Response,Optimal_Outcome
        6501461,57,0,2,0,321,1,0
        5973259,52,0,2,1,648,1,0
        3864522,68,0,2,1,878,1,0
        5405166,71,0,2,1,892,1,0

        5612585,75,1,0,0,6,,1
        3930786,82,1,0,1,100,,1
        3848012,65,1,-1,1,265,,1
        85035058,76,1,0,0,,,1
        3020770,77,1,0,1,,,1
    '''
    with open(csvPath, "w") as file:
        file.write("MRN,Age,OptimalSurgery,ResidualTumor,Censor,TimeSurgeryDeath(d),ChemoResponse,OptimalResult,\n")
        daysPerMonth = 30.4368
        for key in predictDict:
            a = predictDict[key]
            file.write(f"{key},{a['Age']},{a['OptimalSurgery']},{a['ResidualTumor']},None,{int(a['SurvivalMonths']*daysPerMonth)},{a['ChemoResponse']},{a['OptimalResult']},\n")

def readGTDict6Cols(gtPath):
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


def outputPredictDict2Csv6Cols(predictDict, csvPath):
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
            file.write(f"{key},{a['Age']},{a['ResidualTumor']},None,{int(a['SurvivalMonths']*daysPerMonth)},{a['ChemoResponse']},\n")

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



def outputPredictProbDict2Csv(predictProbDict, csvPath):
    '''
        csv data example:
        MRN,Prob1,GT,
        6501461,0.3,0.7,0,

    '''
    with open(csvPath, "w") as file:
        file.write("MRN,Prob1,GT,\n")
        for key in predictProbDict:
            a = predictProbDict[key]
            file.write(f"{key},{a['Prob1']},{a['GT']},\n")
