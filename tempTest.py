
#  test BoundaryLoss
import numpy as np
from DataMgr import DataMgr
from BasicModel import BasicModel

inputSize = (147,281,281)
nDownSamples = 6

xSize = BasicModel.getDownSampleSize(inputSize, nDownSamples);
print(f"bottle neck size: {xSize}")

segSize = BasicModel.getUpSampleSize(xSize, nDownSamples)
print(f"segmentation size: {segSize}")

dataMgr = DataMgr("", "", "")

x = np.array([[3, 0, 0], [0, 0, 0], [5, 6, 0], [0,0,0]])

print(x)

print("after convert")

dataMgr.convertAllZeroSliceToValue(x, -100)
print(x)





# filename =  "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_ROI/72133468B_roi.npy"
# array = np.load(filename)
# print(array.shape)

