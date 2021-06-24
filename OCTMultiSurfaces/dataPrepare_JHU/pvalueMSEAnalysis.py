# Compare model error with student t-test and MSE.
# model 1: YufanHe's model
# model 2: Our model

#JHU method: expJHU_20210501_YufanHe_fullData_A_skm2
#Our method: expJHU_20210507_SurfaceSubnetQ64_fullData_A_skm2


predictPath1 = "/raid001/users/hxie1/data/OCT_JHU/numpy/log/SurfacesUnet_YufanHe_2/expJHU_20210501_YufanHe_fullData_A_skm2/testResult/test/testOutputs.npy"
predictPath2 = "/raid001/users/hxie1/data/OCT_JHU/numpy/log/SurfaceSubnet_Q/expJHU_20210507_SurfaceSubnetQ64_fullData_A_skm2/testResult/test/testOutputs.npy"
gtPath = "/raid001/users/hxie1/data/OCT_JHU/numpy/log/SurfaceSubnet_Q/expJHU_20210507_SurfaceSubnetQ64_fullData_A_skm2/testResult/test/testGts.npy"

model1Name ="JHUYufanHeModel"
model2Name =" OurSurfaceQ64"

hPixelSize =  3.86725 # um

import glob
import numpy as np
import os

import statsmodels.api as sm

print("Compute p-value and MSE:")
print(f"predictPath1= {predictPath1}")
print(f"predictPath2= {predictPath2}")
print(f"gtPath = {gtPath}")
print("============================================================")

predict1All = np.load(predictPath1).astype(float)
predict2All = np.load(predictPath2).astype(float)
gtAll = np.load(gtPath).astype(float)
assert predict1All.shape == gtAll.shape
assert predict2All.shape == gtAll.shape
print(f"segmentation array shape = {gtAll.shape}")  # B,N,W
B,N,W = gtAll.shape

predict1All = np.swapaxes(predict1All,0,1).reshape((N,-1))   # size: NxBxW
predict2All = np.swapaxes(predict2All, 0, 1).reshape((N,-1))
gtAll       = np.swapaxes(gtAll, 0, 1).reshape((N,-1))

model1Error = predict1All -gtAll
model2Error = predict2All -gtAll
print(f"ttest tests the null hypothesis that the population means related to two independent, "
      f"random samples from an approximately normal distribution are equal. ")

# make the output similar with Latex table.
print(f"ttestResult for {N} surfaces signed errors:");
print(f"modleError shape = {model1Error.shape}")
print("\t\t\t testStatistic \t\t pValue \t degreeFreedom")
signedpvalue = []
for n in range(N):
    ttestResult = sm.stats.ttest_ind(model1Error[n,], model2Error[n,])
    print(f"ttestResult for surface {n}: {ttestResult}")
    signedpvalue.append(ttestResult[1])

print(f"==================================")
print(f"p-value for signed errors:")
for n in range(N):
    print(f"{signedpvalue[n]:.4f}\t", end="")
print(f"\n")
print(f"==================================")

print("------------------------------------------------")
print(f"ttestResult for {N} surfaces absolute errors:");
print(f"modleError shape = {model1Error.shape}")
print("\t\t\t testStatistic \t\t pValue \t degreeFreedom")
abspvalue=[]
for n in range(N):
    ttestResult = sm.stats.ttest_ind(np.absolute(model1Error[n,]), np.absolute(model2Error[n,]))
    print(f"ttestResult for surface {n}: {ttestResult}")
    abspvalue.append(ttestResult[1])

print(f"==================================")
print(f"p-value for absolute errors:")
for n in range(N):
    print(f"{abspvalue[n]:.4f}\t", end="")
print(f"\n")
print(f"==================================")


print("------------------------------------------------\n")
print("MSE(predict-gt, 0)= bias^2(predict-gt,0) + variance(predict-gt)")
print(f"\n\n================MSE measure in physical size (um^2) ===============")

mse1= [0,]*N
var1 = [0,]*N
biasSquare1 = [0,]*N
mse2= [0,]*N
var2 = [0,]*N
biasSquare2 = [0,]*N

for n in range(N):
    mse1[n] = sm.tools.eval_measures.mse(predict1All[n]*hPixelSize, gtAll[n]*hPixelSize)
    var1[n]  = np.var(predict1All[n]*hPixelSize-gtAll[n]*hPixelSize)
    biasSquare1[n] = mse1[n] - var1[n]

    mse2[n] = sm.tools.eval_measures.mse(predict2All[n]*hPixelSize, gtAll[n]*hPixelSize)
    var2[n] = np.var(predict2All[n]*hPixelSize-gtAll[n]*hPixelSize)
    biasSquare2[n] = mse2[n]- var2[n]

print(f"in model {model1Name}:")
print(f"MSE\t", end="")
for n in range(N):
    print(f"{mse1[n]:.2f}\t",end="")
print("")
print(f"BiasSquare\t", end="")
for n in range(N):
    print(f"{biasSquare1[n]:.2f}\t",end="")
print("")
print(f"Variance\t", end="")
for n in range(N):
    print(f"{var1[n]:.2f}\t",end="")
print("")

print(f"in model {model2Name}:")
print(f"MSE\t", end="")
for n in range(N):
    print(f"{mse2[n]:.2f}\t",end="")
print("")
print(f"BiasSquare\t", end="")
for n in range(N):
    print(f"{biasSquare2[n]:.2f}\t",end="")
print("")
print(f"Variance\t", end="")
for n in range(N):
    print(f"{var2[n]:.2f}\t",end="")
print("")

print(f"=============END==============")







