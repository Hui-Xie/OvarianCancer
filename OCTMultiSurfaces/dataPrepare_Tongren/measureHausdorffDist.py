
import json
import numpy as np
import os


import sys
sys.path.append(".")
from TongrenFileUtilities import getSurfacesArray

sys.path.append("../..")
from framework.NetTools import  columnHausdorffDist


predictedXmlDir= "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/log/SoftSepar3Unet/expTongren_9Surfaces_SoftConst_20200829A_FixLambda2Unet_CV0/testResult/xml"
gtLabelFile = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/test/surfaces_CV0.npy"
gtIDFile = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/test/patientID_CV0.json"

with open(gtIDFile) as f:
   IDs = json.load(f)

nSlices = len(IDs)

slicesPerVolume = 31

allGT = np.load(gtLabelFile)
B,N,W = allGT.shape

for i in range(0,nSlices, slicesPerVolume):
    # ID: "/home/hxie1/data/OCT_Tongren/control/140009_OD_2602_Volume/20110427064946_OCT01.jpg"
    id = IDs[str(i)]
    volume = os.path.basename(id[0:id.rfind('/')])
    xmlFile = predictedXmlDir+"/"+volume+"_Sequence_Surfaces_Prediction.xml"
    predictS = getSurfacesArray(xmlFile) if i==0 else np.concatenate((predictS, getSurfacesArray(xmlFile)), axis=0)

assert allGT.shape == predictS.shape

columnHausdorffD = columnHausdorffDist(allGT, predictS).reshape((1,N))

print(f"predictedXmlDir =\n{predictedXmlDir}")

print(f"columnHausdorffD = \n{columnHausdorffD}")


print(f"=====End of measure of Hausdorff Distance============")