# compare 2 accuracies along feature dimension

import numpy as np
import matplotlib.pyplot as plt
import os


trainAccuracyFile = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/training/latent/latent_20191210_024607/analyzeImage/accuracyFeature.npy"
testAccuracyFile = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/test/latent/latent_20191210_024607/analyzeImage/accuracyFeature.npy"

outputAnalyzeDir = "/home/hxie1/data/OvarianCancerCT/primaryROI1_1_3/training/latent/latent_20191210_024607/analyzeImage"

F = 1536

trainAccuracy = np.load(trainAccuracyFile)
testAccuracy  = np.load(testAccuracyFile)


indexArray = np.zeros((F,), dtype=np.int)
for i in range(0, F):
    indexArray[i] = i

fig = plt.figure()
plt.xlabel('feature in ascending index')
plt.ylabel('response prediction accuracy from single feature')
plt.ylim([0.4, 0.85])
plt.yscale('linear')
plt.scatter(indexArray, trainAccuracy,s=1)
plt.scatter(indexArray, testAccuracy,s=1)
plt.legend(['training', 'test'])
#plt.show()

plt.savefig(os.path.join(outputAnalyzeDir, f"accurayComparision.png"))

plt.close()
