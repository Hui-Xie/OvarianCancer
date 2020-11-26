# generate all-volume-patient list

srcDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/volumes"

import glob
import os


patientsList = glob.glob(srcDir + f"/*_Volume.npy")
patientsList.sort()
N = len(patientsList)
print(f"total {N} volumes files.")

outputPath = os.path.join(srcDir, f"patientList.txt")
with open(outputPath, "w") as file:
    for i in range(N):
        file.write(f"{patientsList[i]}\n")
print(f"output: {outputPath}")

