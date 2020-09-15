# check whether the nrrd file exist

dataSetTxtPath = "/home/hxie1/data/OvarianCancerCT/survivalPreditData/allMRN.txt"
nrrdPath = "/home/hxie1/data/OvarianCancerCT/rawNrrd/images_1_1_3XYZSpacing"

import os

with open(dataSetTxtPath,'r') as f:
    MRNList = f.readlines()
MRNList = [item[0:-1] for item in MRNList] # erase '\n'
for MRN in MRNList:
    if len(MRN) == 7:
        MRN='0' + MRN
    imagePath = nrrdPath+ f"/{MRN}_CT.nrrd"
    if not os.path.isfile(imagePath):
       print(f"MRN: {MRN} file in not in {nrrdPath}")

print(f"==========Checked total {len(MRNList)} files.=============")

