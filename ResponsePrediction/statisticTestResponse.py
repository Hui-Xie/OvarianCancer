

patientResponsePath = "/home/hxie1/data/OvarianCancerCT/patientResponseDict.json"

# for 20191221 CV Experiment
#testResult0Path = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/LatentCrossValidation/extractLatent_20191210_024607/log/VoteClassifier/latentCV_20191221_Vote_10F_0/testResult_CV0.json"
#outputExcelPath = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/LatentCrossValidation/extractLatent_20191210_024607/log/VoteClassifier/VoteClassifierResult.xls"
#testResult0Path = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/LatentCrossValidation/extractLatent_20191210_024607/log/FCClassifier/latentCV_20191221_FC_10F_0/testResult_CV0.json"
#outputExcelPath = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/LatentCrossValidation/extractLatent_20191210_024607/log/FCClassifier/FCClassifierResult.xls"

# for EXpHighDiceLatentVector_CV_20191223 experiment
# testResult0Path = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/training/latent/extractLatent_20191210_024607/log/VoteClassifier/ExpHighDiceLV_CV_20191223_Vote_10F_0/testResult_CV0.json"
# outputExcelPath = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/training/latent/extractLatent_20191210_024607/log/VoteClassifier/mergeTestResult.xls"

# for ExpFullFeatureLV_CV_20191224 experiment
testResult0Path = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/LatentCrossValidation/rawLatent_20191210_024607/log/FullFeatureVoteClassifier/ExpFullFeatureLV_CV_20191224_10F_0/testResult_CV0.json"
outputExcelPath = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/LatentCrossValidation/rawLatent_20191210_024607/log/FullFeatureVoteClassifier/mergeTestResult.xls"



import json
from xlwt import Workbook
import os

with open(patientResponsePath) as f:
    responseGTDict = json.load(f)

with open(testResult0Path) as f:
    testResult0 = json.load(f)

# merge all test result
testResult = testResult0.copy()
for i in range(1,10): # for 10-fold CV
    testResultiPath = testResult0Path.replace("_10F_0/testResult_CV0.json", f"_10F_{i:d}/testResult_CV{i:d}.json")
    with open(testResultiPath) as f:
        testResulti = json.load(f)
    testResult = {**testResult, **testResulti}

# write excel result
wb = Workbook()
sheet1 = wb.add_sheet("Test Result")
# write table head
sheet1.write(0,0, "PatientID")  # rowIndex, columnIndex, label
sheet1.write(0,1, "TestResult")
sheet1.write(0,2, "GroundTruth")


# write table content
N = len(testResult)
TP=TN=0.0
FP=FN=0.0

row = 1
for key,testResponse in  testResult.items():
    sheet1.write(row,0, key)
    sheet1.write(row,1, testResponse)
    key = key[0:8]
    gtResponse = responseGTDict[key]
    sheet1.write(row,2, gtResponse)
    row +=1

    # compute accuracy, TPR, TNR
    if testResponse == gtResponse:
        if gtResponse ==1:
            TP +=1.0
        else:
            TN +=1.0
    elif gtResponse == 1:
        FN +=1.0
    else:
        FP +=1.0

accuracy = (TP+TN)/(TP+TN+FP+FN)
TPR = TP/(TP+FN)
TNR = TN/(TN+FP)

wb.save(outputExcelPath)
print(f"In {outputExcelPath}:")
print(f"accuracy = {accuracy}, TPR ={TPR}, TNR={TNR} for total N={N} test files 10 folds")

print(f"===========end of statistic test response result=============")


