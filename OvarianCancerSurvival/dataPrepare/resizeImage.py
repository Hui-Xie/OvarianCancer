# Resize Image

nrrdPath = "/home/hxie1/data/OvarianCancerCT/rawNrrd/images_1_1_3XYZSpacing/nrrd"
outputPath = "/home/hxie1/data/OvarianCancerCT/rawNrrd/images_H281_W281"

import os
import SimpleITK as sitk
import glob
import numpy as np
from skimage.transform import resize
dH = 281 # desired H
dW = 281 # desired W

nrrdList = glob.glob(nrrdPath + f"/*_CT.nrrd")

for nrrdFile in nrrdList:
    itkImage = sitk.ReadImage(nrrdFile)
    npVolume = sitk.GetArrayFromImage(itkImage).astype(float)
    S,H,W = npVolume.shape

    resultVolume = np.zeros((S,dH,dW),dtype=float)
    for i in range(S):
        image = npVolume[i,:,:]
        resultVolume[i,:,:] = resize(image, (dH, dW), order=3, anti_aliasing=True)

    fileNameExt = os.path.basename(nrrdFile)
    baseName, ext = os.path.splitext(fileNameExt)
    outFilePath = outputPath+f"/"+baseName+".npy"
    np.save(outFilePath, resultVolume)

print(f"==========END=============")
