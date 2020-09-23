# statistics Nrrd files

dataSetTxtPath = "/home/hxie1/data/OvarianCancerCT/survivalPredict/MRN_179.txt"
nrrdPath = "/home/hxie1/data/OvarianCancerCT/rawNrrd/images_1_1_3XYZSpacing/nrrd"

import os
import SimpleITK as sitk

N = 0
Smin= Smean=Smax = 0 # min, mean, max
Hmin= Hmean=Hmax = 0
Wmin= Wmean=Wmax  =0

with open(dataSetTxtPath, 'r') as f:
    MRNList = f.readlines()
MRNList = [item[0:-1] for item in MRNList]  # erase '\n'
for MRN in MRNList:
    if len(MRN) == 7:
        MRN = '0' + MRN
    imagePath = nrrdPath + f"/{MRN}_CT.nrrd"
    if not os.path.isfile(imagePath):
        print(f"MRN: {MRN} file in not in {nrrdPath}")
    else:
        N +=1
        itkImage = sitk.ReadImage(imagePath)
        npVolume = sitk.GetArrayFromImage(itkImage)
        S, H, W = npVolume.shape
        if 1 == N:
            Smin = Smean = Smax = S  # min, mean, max
            Hmin = Hmean = Hmax = H
            Wmin = Wmean = Wmax = W
        else:
            Smean += S
            Hmean += H
            Wmean += W
            if S < Smin:
                Smin = S
            if H < Hmin:
                Hmin = H
            if W < Wmin:
                Wmin = W
            if S > Smax:
                Smax = S
            if H > Hmax:
                Hmax = H
            if W > Wmax:
                Wmax = W

Smean /=N
Hmean /=N
Wmean /=N
print(f"total {N} nrrd files.")
print(f"Smin= {Smin}, Smean={Smean}, Smax={Smax}")
print(f"Hmin= {Hmin}, Hmean={Hmean}, Hmax={Hmax}")
print(f"Wmin= {Wmin}, Wmean={Wmean}, Wmax={Wmax}")

print(f"==========Checked total {len(MRNList)} files.=============")