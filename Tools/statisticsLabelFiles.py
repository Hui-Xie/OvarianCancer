
# perform statistics of labels file.

import os
import numpy as np


labelDir = "/home/hxie1/data/OvarianCancerCT/primaryROISmall/labels_npy"
suffix = ".npy"

# read all files.
originalCwd = os.getcwd()
os.chdir(labelDir)
filesList = [os.path.abspath(x) for x in os.listdir(labelDir) if suffix in x]
os.chdir(originalCwd)

count0 = 0
count1 = 0
countAll = 0

for file in filesList:
    label3d = np.load(file)
    count1 += np.sum((label3d >0).astype(int))
    countAll += label3d.size

count0 = countAll - count1

print(f"Total {len(filesList)} in {labelDir}")
print(f"0 has {count0} elements, with a rate of  {count0/countAll} ")
print(f"1 has {count1} elements, with a rate of  {count1/countAll} ")
