# convert a csv file to 2D dictionary

csvPath = "/home/hxie1/data/OvarianCancerCT/survivalPredict/8ColsGT/trainingSetGroundTruth.csv"

import csv

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
with open(csvPath, newline='') as csvfile:
    csvList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    csvList = csvList[1:]  # erase table head
    for row in csvList:
        MRN = '0' + row[0] if 7 == len(row[0]) else row[0]
        gtDict[MRN] = {}
        gtDict[MRN]['Age'] = int(row[1])  if 0 != len(row[1]) else -100
        gtDict[MRN]['OptimalSurgery'] =  int(row[2]) if 0 != len(row[2]) else -100
        gtDict[MRN]['ResidualTumor'] = int(row[3]) if 0 != len(row[3]) else -100
        # none data use -100 express
        gtDict[MRN]['Censor'] = int(row[4]) if 0 != len(row[4]) else -100
        gtDict[MRN]['SurvivalMonths'] = int(row[5]) / daysPerMonth if 0 != len(row[5]) else -100
        gtDict[MRN]['ChemoResponse'] = int(row[6]) if 0 != len(row[6]) else -100
        gtDict[MRN]['OptimalResult'] = int(row[7]) if 0 != len(row[7]) else -100


print(f"gtDict has len= {len(gtDict)}")
print(f"============END============")