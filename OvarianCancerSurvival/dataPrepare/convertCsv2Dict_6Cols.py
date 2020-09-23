# convert a csv file to 2D dictionary

csvPath = "/home/hxie1/data/OvarianCancerCT/survivalPredit/trainingSetGroundTruth.csv"

import csv

'''
csv data example:
MRN,Age,ResidualTumor,Censor,TimeSurgeryDeath(d),ChemoResponse
3818299,68,0,1,316,1
5723607,52,0,1,334,0
68145843,70,0,1,406,0
4653841,64,0,1,459,0
96776044,49,0,0,545,1

'''

gtDict= {}

with open(csvPath, newline='') as csvfile:
    csvList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    csvList = csvList[1:]  # erase table head
    for row in csvList:
        lengthRow = len(row)
        MRN = '0'+row[0] if 7==len(row[0]) else row[0]
        gtDict[MRN] = {}
        gtDict[MRN]['Age'] = int(row[1])
        gtDict[MRN]['ResidualTumor'] = int(row[2])
        gtDict[MRN]['Censor'] = int(row[3]) if 0 != len(row[3]) else None
        gtDict[MRN]['SurvivalMonths'] = int(row[4])/30.4368 if 0 != len(row[4]) else None
        gtDict[MRN]['ChemoResponse'] = int(row[5]) if 0 != len(row[5]) else None

print(f"gtDict has len= {len(gtDict)}")
print(f"============END============")