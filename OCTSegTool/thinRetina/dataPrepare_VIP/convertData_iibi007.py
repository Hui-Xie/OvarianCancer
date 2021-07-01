# convert 10 patients data into ground truth images, and numpy array.
# June 29th, Monday, 2021:
# A.  add 3-Bsacn average.
# B.  Add image CLAHE (Contrast Limited Adaptive Histogram Equalization)
# C.  Add 3D-height sampling rate thin-plate spline smoothing on ground truth and prediction.
# D.  add 3 patient as independent test set.

# need python package: simpleitk, scipy, matplotlib, scikit-image

import os
import glob as glob
import SimpleITK as sitk
import json

import matplotlib.pyplot as plt
from utilities import  getSurfacesArray, scaleMatrix
from skimage import exposure  # for CLAHE
# import cv2 as cv  # for CLAHE
import scipy
if scipy.__version__ =="1.7.0":
    from scipy.interpolate import RBFInterpolator  # for scipy 1.7.0
else:
    from scipy.interpolate import Rbf   # for scipy 1.6.2
import random
import numpy as np

extractIndexs = (0, 1, 3, 5, 6, 10) # extracted surface indexes from original 11 surfaces.
surfaceNames =  ("ILM", "RNFL-GCL", "IPL-INL", "OPL-HFL", "BMEIS", "OB_RPE")
pltColors = ('tab:blue', 'tab:orange',  'tab:purple',  'tab:brown',  'tab:red', 'tab:green')
needLegend = True

H = 1024
N = len(extractIndexs)
W = 200  # target image width
C = 1000 # the number of random chosed control points for Thin-Plate-Spline. C is a multiple of 8.


# output Dir:
outputImageDir = "/localscratch/Users/hxie1/data/thinRetina/numpy_13cases/rawGT"
outputNumpyParentDir = "/localscratch/Users/hxie1/data/thinRetina/numpy_13cases"
outputTrainNumpyDir = os.path.join(outputNumpyParentDir, "training")
outputValidationNumpyDir = os.path.join(outputNumpyParentDir, "validation")
outputTestNumpyDir = os.path.join(outputNumpyParentDir, "test")

if not os.path.exists(outputImageDir):
    os.makedirs(outputImageDir)
if not os.path.exists(outputTrainNumpyDir):
    os.makedirs(outputTrainNumpyDir)
if not os.path.exists(outputValidationNumpyDir):
    os.makedirs(outputValidationNumpyDir)
if not os.path.exists(outputTestNumpyDir):
    os.makedirs(outputTestNumpyDir)


# original patientDirList
trainPatientDirList= [  #8 patients
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set1/PVIP2-4060_Macular_200x200_8-25-2009_11-55-11_OD_sn16334_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set1/PVIP2-4073_Macular_200x200_1-3-2013_15-52-39_OS_sn10938_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set1/PVIP2-4084_Macular_512x128_5-14-2012_14-35-40_OD_sn26743_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set1/PVIP2-4081_Macular_512x128_11-11-2010_12-42-15_OS_sn14530_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/Set1/PVIP2-4004_Macular_200x200_10-10-2012_12-17-24_OD_sn11266_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/Set1/PVIP2-4074_Macular_200x200_11-7-2013_8-14-8_OD_sn26558_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/Set1/PVIP2-4088_Macular_512x128_12-4-2012_9-48-42_OD_sn12365_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/Set1/PVIP2-4045_Macular_512x128_4-20-2010_14-18-22_OD_sn12908_cube_z",
]

validationPatientDirList=[  # 2 patients
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set1/PVIP2-4068_Macular_200x200_10-18-2012_12-10-55_OS_sn14463_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/Set1/PVIP2-4083_Macular_200x200_10-24-2012_10-24-46_OS_sn14353_cube_z",
]

testPatientDirList=[
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set2/PVIP2-4093_Macular_200x200_8-28-2013_12-3-48_OS_sn26990_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/Set2/PVIP2-4089_Macular_512x128_3-30-2011_10-5-50_OS_sn17026_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Manual_Correction/Set2/PVIP2-4095_Macular_200x200_3-27-2012_10-17-17_OD_sn18398_cube_z",
]

cases= {
    "training": [trainPatientDirList, outputTrainNumpyDir, 4*(200+128)], # the number is totalSlices.
    "validation": [validationPatientDirList, outputValidationNumpyDir, 2*200],
    "test": [testPatientDirList, outputTestNumpyDir, 2*(200)+128]
}
for datasetName,[patientDirList, outputNumpyDir, totalSlices] in cases.items():
    outputNumpyImagesPath = os.path.join(outputNumpyDir, f"images.npy")  # for smoothed image
    outputNumpyImagesPathClahe = os.path.join(outputNumpyDir, f"images_clahe.npy")  # for CLAHE image
    outputNumpySurfacesPath = os.path.join(outputNumpyDir, f"surfaces.npy")
    outputPatientIDPath = os.path.join(outputNumpyDir, "patientID.json")

    allPatientsImageArray = np.empty((totalSlices , H, W), dtype=float) # for smoothed image
    allPatientsImageArrayClahe = np.empty((totalSlices , H, W), dtype=float) # for CLAHE image
    allPatientsSurfaceArray = np.empty((totalSlices, N, W), dtype=float) # the ground truth of data is float
    patientIDDict = {}


    print(f"Program is outputing raw_GT images in {outputImageDir} for {datasetName}, please wait ......")
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
        npImage = sitk.GetArrayFromImage(itkImage).astype(float)  # in BxHxW dimension
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

        # Average 3 Bscan smoothing.
        smoothedImage = np.zeros_like(npImage,dtype=float)
        for i in range(B):
            i0 = i-1 if i-1>=0 else 0
            i1 = i
            i2 = i+1 if i+1<B else B-1
            smoothedImage[i,] = (npImage[i0, ] +npImage[i1,] +npImage[i2,])/3.0  # intensity in [0,255] in float

        # Use CLAHE (Contrast Limited Adaptive Histogram Equalization) method to increase the contrast of smoothed Bscan.

        # use skimage
        # Rescale image data to range [0, 1]
        smoothedImage = np.clip(smoothedImage,
                          np.percentile(smoothedImage, 1),
                          np.percentile(smoothedImage, 100))
        smoothedImage = (smoothedImage - smoothedImage.min()) / (smoothedImage.max() - smoothedImage.min())
        claheImage = exposure.equalize_adapthist(smoothedImage, kernel_size=[8,64,25], clip_limit=0.01, nbins=256)
        # we need keep the smoothedImage and claheImage

        # npImage = exposure.equalize_adapthist(smoothedImage, kernel_size=[8,64,25], clip_limit=0.01, nbins=256)

        # use opencv, and opencv only suport 2D images.
        # clahe = cv.createCLAHE(clipLimit=40.0, tileGridSize=(64, 25))
        # for i in range(B):
        #    npImage[i,] = clahe.apply(smoothedImage[i,])

        #  a slight smooth the ground truth before using:
        #  A "very gentle" 3D smoothing process (or thin-plate-spline) should be applied to reduce the manual tracing artifact
        #  Check the smoothing results again in the images to make sure they still look reasonable

        # determine the control points of thin-plate-spline
        coordinateSurface = np.mgrid[0:B, 0:W]
        coordinateSurface = coordinateSurface.reshape(2, -1).T  # size (BxW) x2  in 2 dimension.

        # random sample C control points in the original surface of size BxW, with a repeatable random.
        randSeed = 20217  # fix this seed for ground truth and prediction.
        random.seed(randSeed)
        P = list(range(0, B * W))
        chosenList = [0, ] * C
        # use random.sample to choose unique element without replacement.
        chosenList[0:C // 8] = random.sample(P[0:W * B // 4], k=C // 8)
        chosenList[C // 8:C // 2] = random.sample(P[W * B // 4: W * B // 2], k=3 * C // 8)
        chosenList[C // 2:7 * C // 8] = random.sample(P[W * B // 2: W * 3 * B // 4], k=3 * C // 8)
        chosenList[7 * C // 8: C] = random.sample(P[W * 3 * B // 4: W * B], k=C // 8)
        chosenList.sort()
        controlCoordinates = coordinateSurface[chosenList, :]
        for i in range(N):
            surface = surfaces[:, i, :]  # choose surface i, size: BxW
            controlValues = surface.flatten()[chosenList,]
            # for scipy 1.7.0
            if scipy.__version__ =="1.7.0":
                interpolator = RBFInterpolator(controlCoordinates, controlValues, neighbors=None, smoothing=0.0,
                                           kernel='thin_plate_spline', epsilon=None, degree=None)
                surfaces[:, i, :] = interpolator(coordinateSurface).reshape(B, W)
            else:
                # for scipy 1.6.2
                interpolator = Rbf(controlCoordinates[:,0], controlCoordinates[:,1], controlValues, function='thin_plate')
                surfaces[:, i, :] = interpolator(coordinateSurface[:,0], coordinateSurface[:,1]).reshape(B, W)

        # After TPS Interpolation, the surfaces values may exceed the range of [0,H), so it needs clip.
        # for example PVIP2_4045_B128_s127 and s_126 may exceed the low range.
        surfaces = np.clip(surfaces, 0, H-1)

        #  output  numpy array.
        allPatientsImageArray[s:s+B,:,:] = smoothedImage
        allPatientsImageArrayClahe[s:s+B,:,:] = claheImage
        allPatientsSurfaceArray[s:s+B, :, :] = surfaces
        for i in range(B):
            # basename: PVIP2-4074_Macular_200x200_11-7-2013_8-14-8_OD_sn26558_cube_z
            patientIDDict[str(s+i)] = basename + f"_B{B:03d}_s{i:03d}"  #e.g. "_B200_s120"
        s += B

        #  out Raw_GT images
        for i in range(B):
            f = plt.figure(frameon=False)
            DPI = 100
            rowSubplot = 1
            colSubplot = 4
            f.set_size_inches(W * colSubplot / float(DPI), H * rowSubplot / float(DPI))

            plt.margins(0)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

            subplot1 = plt.subplot(rowSubplot, colSubplot, 1)
            subplot1.imshow(npImage[i, :, :], cmap='gray')
            subplot1.axis('off')

            subplot2 = plt.subplot(rowSubplot, colSubplot, 2)
            subplot2.imshow(smoothedImage[i, :, :], cmap='gray')
            subplot2.axis('off')

            subplot3 = plt.subplot(rowSubplot, colSubplot, 3)
            subplot3.imshow(claheImage[i, :, :], cmap='gray')
            subplot3.axis('off')

            subplot4 = plt.subplot(rowSubplot, colSubplot, 4)
            subplot4.imshow(claheImage[i, :, :], cmap='gray')
            for n in range(0, N):
                subplot4.plot(range(0, W), surfaces[i, n, :], pltColors[n], linewidth=1.2)
            if needLegend:
                subplot4.legend(surfaceNames, loc='lower left', ncol=2, fontsize='x-small')
            subplot4.axis('off')

            curImagePath = os.path.join(outputImageDir, basename+f"_B{B:03d}_s{i:03d}_raw_smoothed_clahe_GT.png")

            plt.savefig(curImagePath, dpi='figure', bbox_inches='tight', pad_inches=0)
            plt.close()

    # after reading all patients, save numpy array
    np.save(outputNumpyImagesPath, allPatientsImageArray)
    np.save(outputNumpyImagesPathClahe, allPatientsImageArrayClahe)
    np.save(outputNumpySurfacesPath, allPatientsSurfaceArray)
    with open(outputPatientIDPath, 'w') as fp:
        json.dump(patientIDDict, fp)

print(f"===========END of Convert data==============")
