# change path from c-xwu000 server to iibi007 server

srcPath= "/localscratch/Users/hxie1/data/OCT_Duke/numpy_slices/training/patientList_xwu000.txt"
outputFile = "/localscratch/Users/hxie1/data/OCT_Duke/numpy_slices/training/patientList.txt"

with open(srcPath, 'r') as f:
    patientList = f.readlines()
patientList = [item[0:-1] for item in patientList]

N = len(patientList)
for i in range(N):
    patientList[i] = patientList[i].replace("/home/hxie1/data/", "/localscratch/Users/hxie1/data/")

with open(outputFile, 'w') as f:
    for file in patientList:
        f.write(file + "\n")