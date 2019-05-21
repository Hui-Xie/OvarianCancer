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

# output dictionary
jsonData = json.dumps(IDResponseDict)
f = open(outputResponseDictFile,"w")
f.write(jsonData)
f.close()

print(f"=====================end of program======================")




