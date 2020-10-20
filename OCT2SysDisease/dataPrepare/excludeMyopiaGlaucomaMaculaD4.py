'''
exclude some patients with disease like below:
1. High myopia (Axiallength_26_ormore_exclude$ =1)
2. Glaucoma (Glaucoma_exclude$ =1)
3. Macula or retinal diseases (Retina_exclude$=1)

tag: excludeMGM

'''
import sys
import os

sys.path.append("..")
from network.OCT2SysD_Tools import readBESClinicalCsv

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        print("Usage: command IDfile.csv")
        return -1

    IDPath = sys.argv[1]

    GTPath = "/home/hxie1/data/BES_3K/GTs/BESClinicalGT_Analysis.csv"
    gtLabels = readBESClinicalCsv(GTPath)

    # read ID file
    with open(IDPath, 'r') as idFile:
        IDList = idFile.readlines()
    IDList = [item[0:-1] for item in IDList]  # erase '\n'
    oldLength = len(IDList)

    # delete some IDs with High Myopia, Glaucoma and Macula or retinal diseases
    for ID in IDList:
        labels = gtLabels[int(ID)]
        # 'Axiallength_26_ormore_exclude$', 'Glaucoma_exclude$', 'Retina_exclude$'
        if (1 == labels['Axiallength_26_ormore_exclude$']) or (1 == labels['Glaucoma_exclude$']) or (1 == labels['Retina_exclude$']):
            IDList.remove(ID)
    newLength  = len(IDList)


    # write ID file
    if oldLength - newLength > 0:
        print(f"deleted {oldLength - newLength} IDs in {IDPath}")
        filename, ext = os.path.splitext(IDPath)
        outputFilename = filename +"_excludeMGM" + ext
        with open(outputFilename, "w") as file:
            for id in IDList:
                file.write(f"{id}\n")
    else:
        print("No ID deleted.")

if __name__ == "__main__":
    main()
