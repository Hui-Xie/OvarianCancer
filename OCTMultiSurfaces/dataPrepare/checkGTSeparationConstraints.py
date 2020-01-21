


segDir = "/home/hxie1/data/OCT_Beijing/Correcting_Seg"
volumesDir = "/home/hxie1/data/OCT_Beijing/control"
outputDir = "/home/hxie1/data/OCT_Beijing/markErrorGT"


# glob all file

import glob
import sys
sys.path.append(".")
from FileUtilities import *
import torch
import os


device = torch.device('cuda:1')
# torch.set_printoptions(threshold=10000)

patientsList = glob.glob(segDir + f"/*_Volume_Sequence_Surfaces_Iowa.xml")
outputFile = open(os.path.join(outputDir, "violateSeparation.txt"), "w")


notes1 ="Check whether ground truth conforms the separation constraints: h_{i+1} >= h_i, where i is surface index.\n"
notes2 ="Only check 128:640 columns with column starting index 0 for each paitent and each OCT Bscan.\n"
notes3 ="In output below , column index w mean column (w+128) in original width 768 OCT images.\n"
notes4 ="In output below, bscan index starts with 0 which corresponds OCT1 in the original images.\n "

print(notes1, notes2, notes3, notes4, file=outputFile)


errorPatients = 0
errorNum = 0
for patientXmlPath in patientsList:
    patientSurfaceName = os.path.splitext(os.path.basename(patientXmlPath))[0] # e.g. 1062_OD_9512_Volume_Sequence_Surfaces_Iowa
    patientVolumeName = patientSurfaceName[0:patientSurfaceName.find("_Sequence_Surfaces_Iowa")]  # 1062_OD_9512_Volume

    surfacesArray = getSurfacesArray(patientXmlPath)
    Z, Num_Surfaces, W = surfacesArray.shape
    assert Z == 31 and Num_Surfaces == 11 and W == 768
    surfacesArray = surfacesArray[:,:,128:640]
    surfaces = torch.from_numpy(surfacesArray).to(device)
    surface0 = surfaces[:,:-1,:]
    surface1 = surfaces[:,1:, :]
    if torch.all(surface1 >= surface0):
        continue
    else:
        errorPatients +=1

        errorLocations = torch.nonzero(surface0 > surface1)
        currentErrorNum = errorLocations.shape[0]
        errorNum += currentErrorNum
        outputFile.write(f"\n{patientSurfaceName} violates surface separation constraints in {errorLocations.shape[0]} locations indicated by below coordinates (BScan, surface, width):\n")
        for i in range(currentErrorNum):
            if 0 != i%10 and i!=0:
                outputFile.write(f"[{errorLocations[i,0].item():2d},{errorLocations[i,1].item():2d},{errorLocations[i,2].item():3d}], ")
            else:
                outputFile.write(f"\n[{errorLocations[i,0].item():2d},{errorLocations[i,1].item():2d},{errorLocations[i,2].item():3d}], ")
        outputFile.write("\n")

outputFile.write(f"\n\n=============== {errorPatients} patients with total {errorNum} locations have ground truth not conforming separation constraints ============")
outputFile.close()
print("===========  End of program ===============\n\n")



