# test output predictDict to csv

gtPath = "/home/hxie1/data/OvarianCancerCT/survivalPredict/testSetGroundTruth.csv"

outputPredictPath = "/home/hxie1/data/temp/predictTest.csv"

import sys

sys.path.append("..")
from network.OVTools import *

gtDict = readGTDict(gtPath)

outputPredictDict2Csv(gtDict, outputPredictPath)

print(f"============END=====================")
