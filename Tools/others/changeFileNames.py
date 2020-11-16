# change File names

import glob
import os

srcDir= "/home/hxie1/data/OCT_Duke/numpy_slices/log/SoftSepar3Unet/expDuke_20201113A_FixLambda2Unet/testResult/images"

dataList = glob.glob(srcDir + f"/*.np_npy_GT_Predict.png")
for oldFile in dataList:
    dir, filename = os.path.split(oldFile)
    newFilename = filename.replace(".np_npy_", "_")
    newFile = dir +"/" + newFilename
    os.rename(oldFile, newFile)

print(f"===finished changing filenames in {srcDir}")



