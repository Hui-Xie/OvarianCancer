# show the learned surface separation constraint between Surface 3 and Surface 4 and the ground truth (raw and smoothed)?


predictOutputDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/log/SurfacesUnet/expTongren_9Surfaces_SoftConst_20200813C_CV0/testResult/pretrain"
srcNumpyDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/test"
# only CV0 for this test.
# images_CV0.npy,  surfaces_CV0.npy,  patientID_CV0.json

halfWidth = 15
PadddingMode ="reflect"

testRPath = predictOutputDir + "/testR.npy"
testIDPath = predictOutputDir + "/testID.txt"
imagesPath = srcNumpyDir+ "/images_CV0.npy"
surfacesPath = srcNumpyDir + "/surfaces_CV0.npy"
pltColors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple',  'tab:olive', 'tab:brown', 'tab:pink', 'tab:red', 'tab:cyan', 'tab:blue']


# index = 39 # /home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT09.jpg\n
indexList= [15,39, 46, 77, 108, 139] #15, 39, 46, 77, 108, 139 for compare.
for index in indexList:
    s = 3  # surface N
    sp1 = 4 # surface N+1

    numSlices = 31

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
    outputVisaulRPath = predictOutputDir +f"/{volume}_OCT{sliceID}_GT_R_Predict.png"
    xmlFilePath = predictOutputDir +"/xml"+ f"/{volume}_Sequence_Surfaces_Prediction.xml"
    predictS = getSurfacesArray(xmlFilePath)

    # read ground truth
    image = images[index,]
    surface = surfaces[index,]
    # compute R
    Rgt = surface[1:, :] - surface[0:-1, :] # (N-1)xW
    RgtSmooth = smoothCMA(torch.from_numpy(Rgt), halfWidth, PadddingMode).numpy()
    RPredict = Rs[index,]
    RPredictSmooth = smoothCMA(torch.from_numpy(RPredict), halfWidth, PadddingMode).numpy()
    # compute prediction

    # draw image
    f = plt.figure(frameon=False)
    DPI = f.dpi
    subplotRow = 1
    subplotCol = 3
    f.set_size_inches(W * subplotCol / float(DPI), H * subplotRow / float(DPI))
    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,hspace=0)  # very important for erasing unnecessary margins.

    subplot1 = plt.subplot(subplotRow, subplotCol, 1)
    subplot1.imshow(image, cmap='gray')
    subplot1.plot(range(0, W), surface[s, :].squeeze(), pltColors[s], linewidth=0.9)
    subplot1.plot(range(0, W), surface[sp1, :].squeeze(), pltColors[sp1], linewidth=0.9)
    subplot1.legend([f"g_{s}", f"g_{sp1}"], loc='lower center', ncol=2)
    subplot1.axis('off')

    subplot2 = plt.subplot(subplotRow, subplotCol, 2, facecolor='yellow')
    subplot2.plot(range(0, W), Rgt[s, :].squeeze(), pltColors[0], linewidth=1)
    subplot2.plot(range(0, W), RgtSmooth[s, :].squeeze(), pltColors[1], linewidth=1)
    subplot2.plot(range(0, W), RPredict[s, :].squeeze(), pltColors[2], linewidth=1)
    subplot2.plot(range(0, W), RPredictSmooth[s, :].squeeze(), pltColors[7], linewidth=1)
    # subplot2.set_ylim(0, max(max(Rgt[s, :]), max(RPredict[s,:])))
    subplot2.legend([f"Rgt_{s}", f"RgtSmooth_{s}",f"RPredict_{s}", f"RPredictSmooth_{s}"], loc='lower center', ncol=2)
    subplot2.axis('off')

    subplot3 = plt.subplot(subplotRow, subplotCol, 3)
    subplot3.imshow(image, cmap='gray')
    subplot3.plot(range(0, W), predictS[sliceIndex, s, :].squeeze(), pltColors[s], linewidth=0.9)
    subplot3.plot(range(0, W), predictS[sliceIndex, s, :].squeeze() + RPredict[s,:].squeeze(), pltColors[7], linewidth=0.9)
    subplot3.plot(range(0, W), predictS[sliceIndex, sp1, :].squeeze(), pltColors[sp1], linewidth=0.9)
    subplot3.legend([f"predict_{s}", f"predict_{s}+R_{s}", f"predict_{sp1}"], loc='lower center', ncol=3)
    subplot3.axis('off')

    plt.savefig(outputVisaulRPath, dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()


print("============End==================== ")

