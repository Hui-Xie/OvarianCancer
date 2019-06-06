
from SegDataMgr import SegDataMgr

imagePath = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages"
labelPath = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainLabels"

trainDataMgr = SegDataMgr(imagePath, labelPath, "_CT.nrrd")
trainDataMgr.setDataSize(64, 21,281,281,"TrainData")  #batchSize, depth, height, width
trainDataMgr.setMaxShift(25)                #translation data augmentation
trainDataMgr.setFlipProb(0.3)               #flip data augmentation

pixelStatis = [0, 0, 0, 0]  # the number of pixels labeled as 0, 1,2,3
sliceStatis = [0, 0, 0, 0]  # the number of slices having label 0, 1, 2, 3

print("Start to statistics the label data, please waiting......")

for _, labelsCpu in trainDataMgr.dataLabelGenerator(False):
    (pixelList, sliceList) = trainDataMgr.batchLabelStatistic(labelsCpu, 4)
    pixelStatis = [x + y for x, y in zip(pixelStatis, pixelList)]
    sliceStatis = [x+y for x, y in zip(sliceStatis,sliceList)]

print("in below statistic list,  positions indicate label 0, 1, 2 ,3")

print("pixels statistics: ", pixelStatis)
sumPixels = sum(pixelStatis)
pixelProb = [x/sumPixels for x in pixelStatis]
print("pixels probability: ", pixelProb)
weights = [1/x for x in pixelProb]
print("Label weights in Cross Entropy: ", weights)

print("\n")

print("Slice statistics: ", sliceStatis)
sumSlices = sliceStatis[0]
print("slice probability: ", [x/sumSlices for x in sliceStatis])