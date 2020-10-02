# statistics dataset

gtPath= "/home/hxie1/data/BES_3K/GTs/BESClinicalGT.csv"


keysList=[
  "ID", "Eye", "gender", "Age$", "VA$", "Pres_VA$", "VA_Corr$", "IOP$", "Ref_Equa$", "AxialLength$", "Axiallength_26_ormore_exclude$", \
  "Glaucoma_exclude$", "Retina_exclude$", "Height$", "Weight$", "Waist_Circum$", "Hip_Circum$", "BP_Sys$", "BP_Dia$", "hypertension_bp_plus_history$", \
  "Diabetes$final", "Dyslipidemia_lab$", "Dyslipidemia_lab_plus_his$", "Hyperlipdemia_treat$_WithCompleteZero", "Pulse$", "Cognitive$", \
  "Depression_Correct_wyx", "Drink_quanti_includ0$", "SmokePackYears$", "Glucose$_Corrected2015", "CRPL$_Corrected2015", \
   "Choles$_Corrected2015", "HDL$_Corrected2015", "LDL$_Correcetd2015", "TG$_Corrected2015" ]



import sys
sys.path.append(".")
from readClinicalGT import readBESClinicalCsv

def statisticsData(dataSetIDPath, key="", valueType=None):
    '''

    :param dataSetIDPath:
    :param key:
    :param valueType: binary, number,12binary
    :return:
    '''
    gtDict = readBESClinicalCsv(gtPath)

    with open(dataSetIDPath,'r') as f:
        IDList = f.readlines()
    IDList = [item[0:-1] for item in IDList] # erase '\n'


    minV = -1000
    maxV = -1000
    avgV = 0
    b0Count = 0
    b1Count = 0
    N = 0

    for ID in IDList:
        value = gtDict[int(ID)][key]
        if -100 == float(value):
            continue
        N +=1
        if valueType=="binary":
            value = int(value)
            if 0 == value:
                b0Count +=1
            elif 1 == value:
                b1Count +=1
            else:
                print(f"value= {value} don't match binary type at ID {ID}")
                assert False

        elif valueType == "number":
            value = float(value)
            if 1 ==N:
                minV = value
                maxV = value
                avgV = value
            else:
                minV = value if value < minV else minV
                maxV = value if value > maxV else maxV
                avgV +=value
        elif valueType == "12binary":
            value = int(value)
            if 1 == value:
                b0Count += 1
            elif 2 == value:
                b1Count += 1
            else:
                print(f"value= {value} don't match 12binary type at ID {ID}")
                assert False
        else:
            print("valueType error")
            assert False

    # print result
    print(f"total {len(IDList)} raw IDs in file {dataSetIDPath}")
    print(f"values for {key} have {N} records")
    if (valueType == "binary"):
        print(f"0 rate: {b0Count/N}; 1 rate: {b1Count/N}")
    elif (valueType == "12binary"):
        print(f"1 rate: {b0Count/N}; 2 rate: {b1Count/N}")
    elif valueType == "number":
        avgV /=N
        print(f"min={minV}; mean={avgV}; max={maxV}")
    else:
        print("valueType error")
        assert False
    print("\n")

def printUsage(argv):
    print("============ statistics specific key in a ID dataset  =============")
    print("Usage:")
    print(argv[0], " ID_path  keyName  <binary : 12binary : number>")


def main():

    if len(sys.argv) != 4:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        print(f"keys List = \n{keysList}")
        return -1


    dataSetIDPath = sys.argv[1]
    key = sys.argv[2]
    valueType = sys.argv[3]
    statisticsData(dataSetIDPath, key=key, valueType=valueType)


if __name__ == "__main__":
    main()
