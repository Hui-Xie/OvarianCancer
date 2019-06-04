
#  test BoundaryLoss
import numpy as np

from BasicModel import BasicModel

inputSize = (147,281,281)
nDownSamples = 6

xSize = BasicModel.getDownSampleSize(inputSize, nDownSamples);
print(f"bottle neck size: {xSize}")

segSize = BasicModel.getUpSampleSize(xSize, nDownSamples)
print(f"segmentation size: {segSize}")


# filename =  "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_ROI/72133468B_roi.npy"
# array = np.load(filename)
# print(array.shape)

