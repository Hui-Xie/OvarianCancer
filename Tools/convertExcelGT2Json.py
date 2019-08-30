# read extracted clinical information excel file and save it into json file.

clinicalFile = "/home/hxie1/data/OvarianCancerCT/OptimialResults1st2nd_HuiRevised20190810.xlsx"
outputResponseDictFile = "/home/hxie1/data/OvarianCancerCT/patientResponseDict.json"
outputSurgicalResultDictFile =  "/home/hxie1/data/OvarianCancerCT/patientSurgicalResults.json"
outputTripleResultDictFile =  "/home/hxie1/data/OvarianCancerCT/patientTripleResults.json"

import xlrd
import json

wb = xlrd.open_workbook(clinicalFile)
sheet = wb.sheet_by_index(0)

# read excel file
IDResponseDict = {}
IDSurgicalDict = {}
IDTripleDict   = {}

for i in range(1, sheet.nrows):
    patientID = f"{int(sheet.cell(i, 0).value):08d}"
    if patientID in IDResponseDict.keys():
        print(f"{patientID} repeated in the input file")
    else:
        IDResponseDict[patientID] = sheet.cell(i, 5).value
        IDSurgicalDict[patientID] = sheet.cell(i, 1).value
        IDTripleDict[patientID] = [sheet.cell(i, 1).value, sheet.cell(i, 3).value, sheet.cell(i, 4).value] # surgery, chemo, survival

print(f"Program get {len(IDResponseDict)} patients data for OptimalOutcome.")
print(f"Program get {len(IDSurgicalDict)} patients data for Surgical Outcome.")
print(f"Program get {len(IDTripleDict)}   patients data for Triple Outcome.")

# output response dictionary
jsonData = json.dumps(IDResponseDict)
f = open(outputResponseDictFile,"w")
f.write(jsonData)
f.close()

# output Surgical dictionary
jsonData = json.dumps(IDSurgicalDict)
f = open(outputSurgicalResultDictFile,"w")
f.write(jsonData)
f.close()

# output Triple dicionary
jsonData = json.dumps(IDTripleDict)
f = open(outputTripleResultDictFile,"w")
f.write(jsonData)
f.close()

print(f"=====================end of program======================")