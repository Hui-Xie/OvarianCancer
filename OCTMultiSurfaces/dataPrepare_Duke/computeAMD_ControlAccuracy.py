# compute AMD and control group accuracy separately

#predictDir ="/home/hxie1/data/OCT_Duke/numpy_slices/log/SoftSepar3Unet/expDuke_20201113A_FixLambda2Unet/testResult/xml"
predictDir = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/xml"
gtDir = "/home/hxie1/data/OCT_Duke/numpy_slices/test"

N = 3
W = 361
B = 51  #Bscan number for each volume
hPixelSize =  3.24

import glob
import numpy as np
import os
import torch
import sys
sys.path.append("..")
from dataPrepare_Tongren.TongrenFileUtilities import getSurfacesArray
from network.OCTOptimization import computeErrorStdMuOverPatientDimMean
sys.path.append("../..")
from framework.NetTools import  columnHausdorffDist


AMDXmlList = glob.glob(predictDir + f"/AMD_*_images_Sequence_Surfaces_Prediction.xml")
ControlXmlList = glob.glob(predictDir + f"/Control_*_images_Sequence_Surfaces_Prediction.xml")

print("Compute performance in AMD and Control group separately")
print(f"predictDir= {predictDir}")
print(f"gtDir = {gtDir}")
print("===============")


twoGroupDict = {"AMD":AMDXmlList, "Control":ControlXmlList}
for groupName, xmlList in twoGroupDict.items():
    Num = len(xmlList)
    predictAll = np.empty([Num*B,N,W])
    gtAll = np.empty([Num*B,N, W])
    i = 0
    for xmlPath in xmlList:
        volumeSurfaces = getSurfacesArray(xmlPath)
        predictAll[i:i+B,] = volumeSurfaces
        _, stemname = os.path.split(xmlPath)
        volumeName = stemname[0: stemname.find("_images_Sequence_Surfaces_Prediction.xml")]
        gtSliceList = glob.glob(gtDir +f"/{volumeName}_surfaces_s[0-5][0-9].npy")
        assert len(gtSliceList) == B
        gtSliceList.sort()
        for slicePath in gtSliceList:
            bscanSurface = np.load(slicePath)
            gtAll[i,] = bscanSurface
            i = i+1

    assert predictAll.shape == gtAll.shape
    # use numpy for computing Hausdorff distance
    columnHausdorffD = columnHausdorffDist(predictAll, gtAll).reshape((1, N))
    # use torch.tensor for segmenation accuracy
    predictAll = torch.from_numpy(predictAll)
    gtAll = torch.from_numpy(gtAll)
    stdSurfaceError, muSurfaceError, stdError, muError = computeErrorStdMuOverPatientDimMean(predictAll, gtAll,
                                                          slicesPerPatient=B,  hPixelSize=hPixelSize, goodBScansInGtOrder=None)

    print(f"GroupName: {groupName}")
    print(f"case number = {Num}")
    print(f"stdSurfaceError = {stdSurfaceError}")
    print(f"muSurfaceError = {muSurfaceError}")
    print(f"HausdorffDistance in pixel = {columnHausdorffD}")
    print(f"HausdorffDistance in physical size (micrometer) = {columnHausdorffD*hPixelSize}")
    print(f"stdError = {stdError}")
    print(f"muError = {muError}")
    print(f"===============")







