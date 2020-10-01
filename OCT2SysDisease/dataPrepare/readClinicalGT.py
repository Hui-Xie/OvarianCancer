
# read BES clinical data

gtPath= "/home/hxie1/data/BES_3K/GTs/BESClinicalGT.csv"
import csv

def readBESClinicalCsv(gtPath):
    '''
        ID,Eye,gender,Age$,VA$,Pres_VA$,VA_Corr$,IOP$,Ref_Equa$,AxialLength$,Axiallength_26_ormore_exclude$,Glaucoma_exclude$,Retina_exclude$,Height$,Weight$,Waist_Circum$,Hip_Circum$,BP_Sys$,BP_Dia$,hypertension_bp_plus_history$,Diabetes$final,Dyslipidemia_lab$,Dyslipidemia_lab_plus_his$,Hyperlipdemia_treat$_WithCompleteZero,Pulse$,Cognitive$,Depression_Correct_wyx,Drink_quanti_includ0$,SmokePackYears$,Glucose$_Corrected2015,CRPL$_Corrected2015,Choles$_Corrected2015,HDL$_Corrected2015,LDL$_Correcetd2015,TG$_Corrected2015
2.0,1,2,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
3.0,1,1,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
5.0,1,1,53,0.6,0.6,1,12,-1,25.14,,,,186,108,118,110,135,70,1,0,0,0,0,75,30,24.44,0.5,1,5.15,7.2,5.63,0.99,4.47,1.36
6.0,1,2,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
7.0,1,2,53,0.8,0.8,1,18,0.75,21.57,,,,164,72,102,109,143,82,1,0,1,1,1,58,22,31.11,0,0,5.25,0.79,5.8,1.24,4.49,1.9
8.0,1,1,57,0.6,0.6,0.8,18,-0.38,23.84,,,,165,80,105,107,132,65,1,1,0,0,0,93,23,23.33,1,5.25,14.37,0.01,4.29,1.18,3.07,1.56
9.0,1,2,60,0.6,0.6,1,15,0.38,22.48,,,,156,74,103,116,140,86,0,0,0,0,0,69,27,23.33,0,2,5.04,0.28,4.97,1.65,3.26,1.09
10.0,1,2,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
13.0,1,1,60,0.5,0.5,0.8,11,-1.38,23.14,,,,167.5,57,80,91,104,69,0,0,0,0,0,72,30,20,2,4.5,4.19,0.21,3.22,1.93,1.48,0.42


    '''
    gtDict = {}
    with open(gtPath, newline='') as csvfile:
        csvList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
        tableHead = csvList[0]
        nCols = len(tableHead)
        csvList = csvList[1:]  # erase table head
        for row in csvList:
            if len(row[0])>0:
                ID = int(float(row[0]))
                gtDict[ID] = {}
                for c in range(1, nCols):
                    key = tableHead[c]
                    value = row[c] if 0 != len(row[c]) else -100
                    gtDict[ID][key] = value
    return gtDict


def main():
    gtDict = readBESClinicalCsv(gtPath)
    print(f"gtDict length = {len(gtDict)}")
    print(f"gtDict[13] = {gtDict[13]}")

if __name__ == "__main__":
    main()