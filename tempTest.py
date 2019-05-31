
#  test BoundaryLoss
import numpy as np

from Image3dPredictModel import Image3dPredictModel

model = Image3dPredictModel(12, 2, inputSize =(73,141,141), nDownSample=5)

xSize = model.getBottleNeckSize()

print(xSize)


# filename =  "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages_ROI/72133468B_roi.npy"
# array = np.load(filename)
# print(array.shape)

