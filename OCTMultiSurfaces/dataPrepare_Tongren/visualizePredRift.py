
predictOutputDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/log/SurfacesUnet/expTongren_9Surfaces_SoftConst_20200808_CV0_8Grad_LearnPair_Pretrain_LR1/testResult/realtime"
srcNumpyDir = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/test"
# only CV0 for this test.
#  images_CV0.npy,  surfaces_CV0.npy,  patientID_CV0.json

testRPath = predictOutputDir + "/testR.npy"

numPatients = 8
numSlices = 31

import numpy as np
import os
import matplotlib.pyplot as plt
import glob


