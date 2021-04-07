# Compare model error with student t-test and MSE.
# model 1: pure surface branch
# model 2: SoftSeparation model 2 with gradient paitwise term.


predictDir1 = "/localscratch/Users/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet_M/expDuke_20210322_SurfaceSubnet_M_iibi007/testResult/xml"
predictDir2 = "/localscratch/Users/hxie1/data/OCT_Duke/numpy_slices/log/SoftSeparationNet_A/expDuke_SoftSeparation_A_20210329_iibi007/testResult/xml"
gtDir = "/localscratch/Users/hxie1/data/OCT_Duke/numpy_slices/test"

model1Name ="       PureSurfaceModel"
model2Name ="NxWLambdaSoftSeparation"

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
sys.path.append("../..")
from framework.NetTools import  columnHausdorffDist
import statsmodels.api as sm

AMDXmlList1 = glob.glob(predictDir1 + f"/AMD_*_images_Sequence_Surfaces_Prediction.xml")
ControlXmlList1 = glob.glob(predictDir1 + f"/Control_*_images_Sequence_Surfaces_Prediction.xml")

# the xml file list of model 2 uses replacing dir1 with dir2 to get.
#AMDXmlList2 = glob.glob(predictDir2 + f"/AMD_*_images_Sequence_Surfaces_Prediction.xml")
#ControlXmlList2 = glob.glob(predictDir2 + f"/Control_*_images_Sequence_Surfaces_Prediction.xml")

print("Compare model accuracy in AMD and Control group separately")
print(f"predictDir1= {predictDir1}")
print(f"predictDir2= {predictDir2}")
print(f"gtDir = {gtDir}")
print("============================================================")

twoGroupDict = {"AMD":AMDXmlList1, "Control":ControlXmlList1}
for groupName, xmlList in twoGroupDict.items():
    Num = len(xmlList)
    print(f"GroupName: {groupName}")
    print(f"case number = {Num}")

    predict1All = np.empty([Num*B,N,W])
    predict2All = np.empty([Num*B,N,W])
    gtAll = np.empty([Num*B,N, W])
    i = 0
    for xmlPath1 in xmlList:
        xmlPath2 = xmlPath1.replace(predictDir1, predictDir2)
        predict1All[i:i + B, ] = getSurfacesArray(xmlPath1)
        predict2All[i:i + B, ] = getSurfacesArray(xmlPath2)

        _, stemname = os.path.split(xmlPath1)
        volumeName = stemname[0: stemname.find("_images_Sequence_Surfaces_Prediction.xml")]
        gtSliceList = glob.glob(gtDir +f"/{volumeName}_surfaces_s[0-5][0-9].npy")
        assert len(gtSliceList) == B
        gtSliceList.sort()
        for slicePath in gtSliceList:
            gtAll[i,] = np.load(slicePath)
            i = i+1

    assert predict1All.shape == gtAll.shape
    assert predict2All.shape == gtAll.shape

    predict1All = np.swapaxes(predict1All,0,1).reshape((N,-1))   # size: Nx(NumxB)xW -> Nx(NumxBxW)
    predict2All = np.swapaxes(predict2All, 0, 1).reshape((N,-1))
    gtAll       = np.swapaxes(gtAll, 0, 1).reshape((N,-1))

    model1Error = predict1All -gtAll
    model2Error = predict2All -gtAll
    print(f"ttest tests the null hypothesis that the population means related to two independent, "
          f"random samples from an approximately normal distribution are equal. ")

    print(f"ttestResult for {N} surfaces signed errors:");
    print(f"modleError shape = {model1Error.shape}")
    print("\t\t\t testStatistic \t\t pValue \t degreeFreedom")
    for n in range(N):
        ttestResult = sm.stats.ttest_ind(model1Error[n,], model2Error[n,])
        print(f"ttestResult for surface {n}: {ttestResult}")

    print("------------------------------------------------")
    print(f"ttestResult for {N} surfaces absolute errors:");
    print(f"modleError shape = {model1Error.shape}")
    print("\t\t\t testStatistic \t\t pValue \t degreeFreedom")
    for n in range(N):
        ttestResult = sm.stats.ttest_ind(np.absolute(model1Error[n,]), np.absolute(model2Error[n,]))
        print(f"ttestResult for surface {n}: {ttestResult}")


    # MSE(predict-gt)= bias^2(predict, gt) + variance(predict)
    print(f"\n\n================MSE measure===============")

    print(f"\t\t\t\t MSE \t\t BiasSquare \t\t Variance ")
    for n in range(N):
        mse1 = sm.tools.eval_measures.mse(predict1All[n], gtAll[n])
        var1  = np.var(predict1All[n]-gtAll[n])
        biasSquare1 = mse1 - var1

        mse2 = sm.tools.eval_measures.mse(predict2All[n], gtAll[n])
        var2 = np.var(predict2All[n]-gtAll[n])
        biasSquare2 = mse2 - var2
        print(f"surface {n} in {model1Name}:\t {mse1}\t{biasSquare1}\t{var1}")
        print(f"surface {n} in {model2Name}:\t {mse2}\t{biasSquare2}\t{var2}")
        print(f"-----------------------------------")
    print(f"=======================================")







