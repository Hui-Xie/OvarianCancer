# convert 10 patients data into ground truth images, and numpy array.

import os
import glob as glob
import sys
import SimpleITK as sitk
import json
import matplotlib.pyplot as plt
from utilities import  getSurfacesArray, scaleMatrix, get3PointSmoothMatrix

import numpy as np

extractIndexs = (0, 1, 3, 5, 6, 10) # extracted surface indexes from original 11 surfaces.
surfaceNames =  ("ILM", "RNFL-GCL", "IPL-INL", "OPL-HFL", "BMEIS", "OB_RPE")
pltColors = ('tab:blue', 'tab:orange',  'tab:purple',  'tab:brown',  'tab:red', 'tab:green')
needLegend = True

H = 1024
N = len(extractIndexs)
W = 200  # target image width

# output Dir:
outputImageDir = "/home/hxie1/data/Ophthalmology/thinRetina/rawGT"
outputNumpyParentDir = "/home/hxie1/data/Ophthalmology/thinRetina/numpy"
outputTrainNumpyDir = os.path.join(outputNumpyParentDir, "training")
outputTestNumpyDir = os.path.join(outputNumpyParentDir, "test")

# original patientDirList
trainPatientDirList= [  #8 patients
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/PVIP2-4060_Macular_200x200_8-25-2009_11-55-11_OD_sn16334_cube_z",
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/PVIP2-4073_Macular_200x200_1-3-2013_15-52-39_OS_sn10938_cube_z",
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/PVIP2-4084_Macular_512x128_5-14-2012_14-35-40_OD_sn26743_cube_z",
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/PVIP2-4081_Macular_512x128_11-11-2010_12-42-15_OS_sn14530_cube_z",
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/PVIP2-4004_Macular_200x200_10-10-2012_12-17-24_OD_sn11266_cube_z",
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/PVIP2-4074_Macular_200x200_11-7-2013_8-14-8_OD_sn26558_cube_z",
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/PVIP2-4088_Macular_512x128_12-4-2012_9-48-42_OD_sn12365_cube_z",
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/PVIP2-4045_Macular_512x128_4-20-2010_14-18-22_OD_sn12908_cube_z",
]

testPatientDirList=[  # 2 patients
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/PVIP2-4068_Macular_200x200_10-18-2012_12-10-55_OS_sn14463_cube_z",
"/home/hxie1/data/garvinlab/Data/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/PVIP2-4083_Macular_200x200_10-24-2012_10-24-46_OS_sn14353_cube_z",
]

trainingCase = False

if  trainingCase:
    patientDirList = trainPatientDirList
    outputNumpyDir = outputTrainNumpyDir
    totalSlices =  4*(200+128)
else:
    patientDirList = testPatientDirList
    outputNumpyDir = outputTestNumpyDir
    totalSlices = 2*200

outputNumpyImagesPath = os.path.join(outputNumpyDir, f"images.npy")
outputNumpySurfacesPath = os.path.join(outputNumpyDir, f"surfaces.npy")
outputPatientIDPath = os.path.join(outputNumpyDir, "patientID.json")

allPatientsImageArray = np.empty((totalSlices , H, W), dtype=float)
allPatientsSurfaceArray = np.empty((totalSlices, N, W), dtype=float) # the ground truth of JHU data is float
patientIDDict = {}


print(f"Program is outputing raw_GT images in {outputImageDir}, please wait ......")
s = 0 # initial slice index
for patientDir in patientDirList:
    # get volumePath and surfacesXmlPath
    octVolumeFileList = glob.glob(patientDir + f"/*_OCT_Iowa.mhd")
    assert len(octVolumeFileList) == 1
    octVolumePath = octVolumeFileList[0]
    dirname = os.path.dirname(octVolumePath)
    basename = os.path.basename(octVolumePath)
    basename = basename[0:basename.rfind("_OCT_Iowa.mhd")]
    surfacesXmlPath = os.path.join(dirname, basename+f"_Surfaces_Iowa_Ray.xml")
    if not os.path.isfile(surfacesXmlPath):
        surfacesXmlPath = os.path.join(dirname, basename+f"_Surfaces_Iowa.xml")
        if not os.path.isfile(surfacesXmlPath):
            print("Error: can not find surface xml file")
            assert False

    #  convert Ray's special raw format to standard BxHxW for image, and BxSxW format for surface.
    #  Ray mhd format in BxHxW dimension, but it flip the H and W dimension.
    #  for 200x1024x200 image, and 128x1024x512 in BxHxW direction.
    itkImage = sitk.ReadImage(octVolumePath)
    npImage = sitk.GetArrayFromImage(itkImage)  # in BxHxW dimension
    npImage = np.flip(npImage, (1, 2))  # as ray's format filp H and W dimension.
    B,curH,curW = npImage.shape
    assert H == curH

    surfaces = getSurfacesArray(surfacesXmlPath)  # size: SxNxW, where N is number of surfacres.
    surfaces = surfaces[:, extractIndexs, :]   #  extract 6 surfaces (0, 1, 3, 5, 6, 10)
    # its surface names: ["ILM", "RNFL-GCL", "IPL-INL", "OPL-HFL", "BMEIS", "OB_RPE"]
    B1, curN, _ = surfaces.shape
    assert N == curN
    assert B == B1

    #  scale down image and surface, if W = 512.
    if npImage.shape == (128, 1024, 512):  # scale image to 1024x200.
        scaleM = scaleMatrix(B, curW, W)
        npImage = np.matmul(npImage, scaleM)
        surfaces = np.matmul(surfaces, scaleM)
    else:
        assert curW == W

    #  flip all OS eyes into OD eyes
    if "_OS_" in basename:
        npImage = np.flip(npImage, 2)
        surfaces = np.flip(surfaces, 2)

    # Make sure alll surfaces not interleave, especially the top surface of GCIPL (i.e., surface_1) is NOT above ILM (surface_0)
    for i in range(1, N):
        surfaces[:, i, :] = np.where(surfaces[:, i, :] < surfaces[:, i - 1, :], surfaces[:, i - 1, :],
                                     surfaces[:, i, :])

    #  a slight smooth the ground truth before using:
    #  A "very gentle" 3D smoothing process (or thin-plate-spline) should be applied to reduce the manual tracing artifact
    #  Check the smoothing results again in the images to make sure they still look reasonable
    smoothM = get3PointSmoothMatrix(B,W)
    surfaces = np.matmul(surfaces, smoothM)

    #  output  numpy array.
    allPatientsImageArray[s:s+B,:,:] = npImage
    allPatientsSurfaceArray[s:s+B, :, :] = surfaces
    for i in range(B):
        # basename: PVIP2-4074_Macular_200x200_11-7-2013_8-14-8_OD_sn26558_cube_z
        patientIDDict[str(s+i)] = basename + f"_s{i:03d}"
    s += B

    #  out Raw_GT images
    for i in range(B):
        f = plt.figure(frameon=False)
        DPI = 100
        rowSubplot = 1
        colSubplot = 2
        f.set_size_inches(W * colSubplot / float(DPI), H * rowSubplot / float(DPI))

        plt.margins(0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

        subplot1 = plt.subplot(rowSubplot, colSubplot, 1)
        subplot1.imshow(npImage[i, :, :], cmap='gray')
        subplot1.axis('off')

        subplot2 = plt.subplot(rowSubplot, colSubplot, 2)
        subplot2.imshow(npImage[i, :, :], cmap='gray')
        for n in range(0, N):
            subplot2.plot(range(0, W), surfaces[i, n, :], pltColors[n], linewidth=1.2)
        if needLegend:
            subplot2.legend(surfaceNames, loc='lower left', ncol=2, fontsize='x-small')
        subplot2.axis('off')

        curImagePath = os.path.join(outputImageDir, basename+f"_s{i:03d}_raw_GT.png")

        plt.savefig(curImagePath, dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close()

# after reading all patients, save numpy array
np.save(outputNumpyImagesPath, allPatientsImageArray)
np.save(outputNumpySurfacesPath, allPatientsSurfaceArray)
with open(outputPatientIDPath, 'w') as fp:
    json.dump(patientIDDict, fp)

print(f"===========END of Convert data==============")
