# draw the relation between lambda and meanError

csvPath = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SearchLambda2Unet/expDuke_20201109A_SearchLambda2Unet/testResult/searchLambda_replaceRwithGT_0.csv"

import matplotlib.pyplot as plt
import os
import numpy as np

outputDir, outputBasename = os.path.split(csvPath)
outputBasename, _ = os.path.splitext(outputBasename)


# read csv file
import csv
with open(csvPath, newline='') as csvfile:
    csvList = list(csv.reader(csvfile, delimiter=',', quotechar='|'))
    csvList = csvList[1:]
    csvArray = np.asarray(csvList)
    meanError = csvArray[:,1].astype(np.float)
    lambdaValue = csvArray[:,3].astype(np.float)

plt.plot(lambdaValue, meanError)
plt.xlabel('lambda')
plt.ylabel('meanError(micrometer)')
N = 10.0  # the number of ticks in axis
xticks = list(np.arange(0.0,np.max(lambdaValue)*1.2, np.max(lambdaValue)/5))
plt.xticks(xticks)
yticks = list(np.arange(np.min(meanError),np.max(meanError), (np.max(meanError)-np.min(meanError))/N))
plt.yticks(yticks)
plt.title('Grid Search Lambda')
plt.savefig(os.path.join(outputDir,outputBasename + "_gridSearch.png"), dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()
print(f"=============End of csv file read==============")
# draw curve



