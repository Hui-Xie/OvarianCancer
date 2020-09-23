# test output predictDict to csv

gtPath = "/home/hxie1/data/OvarianCancerCT/survivalPredict/testSetGroundTruth.csv"

outputPredictPath = "/home/hxie1/data/temp/predictTest.csv"

import sys

sys.path.append("..")
from network.OVTools import *

gtDict = readGTDict8Cols(gtPath)

outputPredictDict2Csv8Cols(gtDict, outputPredictPath)

print(f"============END=====================")
