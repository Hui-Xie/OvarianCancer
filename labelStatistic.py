
from DataMgr import DataMgr

imagePath = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainImages"
labelPath = "/home/hxie1/data/OvarianCancerCT/Extract_uniform/trainLabels"

trainDataMgr = DataMgr(imagePath, labelPath)
trainDataMgr.setDataSize(64, 21,281,281,4)  #batchSize, depth, height, width, k
trainDataMgr.setMaxShift(25)                #translation data augmentation
trainDataMgr.setFlipProb(0.3)               #flip data augmentation

labelStatis = [0, 0, 0, 0]  # the number of pixels labeled as 0, 1,2,3
sliceStatis = [0, 0, 0, 0]  # the number of slices having label 0, 1, 2, 3

print("Start to statistics the label data, please waiting......")

for _, labelsCpu in trainDataMgr.dataLabelGenerator(False):
    (labelList, sliceList) = trainDataMgr.batchLabelStatistic(labelsCpu, 4)
    labelStatis = [x+y for x, y in zip(labelStatis,labelList)]
    sliceStatis = [x+y for x, y in zip(sliceStatis,sliceList)]

print("Label statistics: ", labelStatis)
print("Slice statistics: ", sliceStatis)


