# read extracted clinical information excel file and save it into json file.

clinicalFile = "/home/hxie1/data/OvarianCancerCT/OptimialResults1st2nd_HuiRevised20190810.xlsx"
outputResponseDictFile = "/home/hxie1/data/OvarianCancerCT/patientResponseDict.json"

import xlrd
import json

wb = xlrd.open_workbook(clinicalFile)
sheet = wb.sheet_by_index(0)

# read excel file
IDResponseDict = {}
for i in range(1, sheet.nrows):
    patientID = f"{int(sheet.cell(i, 0).value):08d}"
    if patientID in IDResponseDict.keys():
        print(f"{patientID} repeated in the input file")
    else:
        IDResponseDict[patientID] = sheet.cell(i, 5).value

print(f"Program get {len(IDResponseDict)} patients data for optimalOutcome.")

# output dictionary
jsonData = json.dumps(IDResponseDict)
f = open(outputResponseDictFile,"w")
f.write(jsonData)
f.close()

print(f"=====================end of program======================")