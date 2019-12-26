# read clinical information and save it into json file.

clinicalFile = "/home/hxie1/data/OvarianCancerCT/outcomes_CT_Doug.xlsx"
outputResponseDictFile = "/home/hxie1/data/OvarianCancerCT/patientResponseDict.json"

import xlrd
import json

wb = xlrd.open_workbook(clinicalFile)
sheet = wb.sheet_by_index(0)

# read excel file
IDResponseDict = {}
nDiscards = 0
for i in range(1, sheet.nrows):
    patientID = f"{int(sheet.cell(i, 2).value):08d}"
    optimalOutcome = sheet.cell(i, sheet.ncols-1).value
    if optimalOutcome == "Yes":
        optimalOutcome = 1
    elif optimalOutcome == "No":
        optimalOutcome = 0
    else:
        nDiscards += 1
        continue
    IDResponseDict[patientID] = optimalOutcome

# output information
print(f"Program discards {nDiscards} patients data as there are no optimialOutcome for them.")
print(f"Program get {len(IDResponseDict)} patients data for optimalOutcome.")

# statistics
train1 = 0
train0 = 0
test1 = 0
test0 = 0
for patientID in IDResponseDict.keys():
    if int(patientID) <=88032071: # train data
        if IDResponseDict[patientID] == 1:
            train1 += 1
        else:
            train0 += 1
    else:                         # test data
        if IDResponseDict[patientID] == 1:
            test1 += 1
        else:
            test0 += 1

print(f"In training set, {train1} 1s, and {train0} 0s")
print(f"In     test set, {test1}  1s, and {test0}  0s")
print(f"total {len(IDResponseDict)} patients")

# output dictionary
jsonData = json.dumps(IDResponseDict)
f = open(outputResponseDictFile,"w")
f.write(jsonData)
f.close()

print(f"=====================end of program======================")




