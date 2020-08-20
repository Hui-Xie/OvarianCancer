# show the learned surface separation constraint and the ground truth (raw and smoothed)


predictOutputDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/log/RiftSubnet/expTongren_9Surfaces_SoftConst_20200819A_RiftSubnet_CV0/testResult"
visualRPath = predictOutputDir+"/visualR"
srcNumpyDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/test"
# only CV0 for this test.
# images_CV0.npy,  surfaces_CV0.npy,  patientID_CV0.json

halfWidth = 15
numSlices = 31
PadddingMode ="reflect"

testRPath = predictOutputDir + "/testR.npy"
testIDPath = predictOutputDir + "/testID.txt"
imagesPath = srcNumpyDir+ "/images_CV0.npy"
surfacesPath = srcNumpyDir + "/surfaces_CV0.npy"
pltColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',  'tab:olive', 'tab:brown', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:blue']


import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append("../network")
from OCTAugmentation import smoothCMA
import torch
sys.path.append(".")
from TongrenFileUtilities import getSurfacesArray

if not os.path.exists(visualRPath):
    os.makedirs(visualRPath)

# index = 39 # /home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT09.jpg\n
indexList= [15,39, 46, 77, 108, 139] #15, 39, 46, 77, 108, 139 for compare.
for index in indexList:
    # s = 3  # surface N
    # sp1 = 4 # surface N+1

    # find file
    with open(testIDPath, 'r') as f:
        IDList = f.readlines()
    IDList = [item[0:-1] for item in IDList]  #erase tail "\n"

    images = np.load(imagesPath)  # BxHxW
    B,H,W = images.shape
    surfaces = np.load(surfacesPath) #BxNxW
    N = surfaces.shape[1]
    Rs = np.load(testRPath)

    a = IDList[index] # srcImagePath
    volume = a[a.find("control/")+8: a.rfind("/")]
    sliceID = a[a.rfind("_OCT")+4: a.rfind(".")]
    sliceIndex = int(sliceID)-1
    outputVisaulRPath = visualRPath +f"/{volume}_OCT{sliceID}_rawR_blue_smoothR_green_PredictR_red.png"
    # xmlFilePath = predictOutputDir +"/xml"+ f"/{volume}_Sequence_Surfaces_Prediction.xml"
    # predictS = getSurfacesArray(xmlFilePath)

    # read ground truth
    image = images[index,]
    surface = surfaces[index,]
    # compute R
    Rgt = surface[1:, :] - surface[0:-1, :] # (N-1)xW
    RgtSmooth = smoothCMA(torch.from_numpy(Rgt), halfWidth, PadddingMode).numpy()
    RPredict = Rs[index,]
    Height = max(np.amax(Rgt), np.amax(RgtSmooth), np.amax(RPredict))
    # RPredictSmooth = smoothCMA(torch.from_numpy(RPredict), halfWidth, PadddingMode).numpy()
    # compute prediction

    # draw image
    f = plt.figure(frameon=False)
    DPI = f.dpi
    subplotRow = Rs.shape[1]
    subplotCol = 1
    f.set_size_inches(W * subplotCol / float(DPI), Height * subplotRow / float(DPI))
    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

    for s in range(0, N-1):
        subplots = plt.subplot(subplotRow, subplotCol, s+1)
        subplots.plot(range(0, W), Rgt[s, :].squeeze(), 'tab:blue', linewidth=1)
        subplots.plot(range(0, W), RgtSmooth[s, :].squeeze(), 'tab:green', linewidth=1)
        subplots.plot(range(0, W), RPredict[s, :].squeeze(), 'tab:red', linewidth=1)
        subplots.axis('off')

    plt.savefig(outputVisaulRPath, dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()


print("============End==================== ")

