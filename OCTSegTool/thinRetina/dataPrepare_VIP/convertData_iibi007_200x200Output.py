# convert 19 patients data into ground truth images, and numpy array.
# June 29th, Monday, 2021:
# A.  add 3-Bsacn average.
# B.  Add image CLAHE (Contrast Limited Adaptive Histogram Equalization)
# C.  Add 3D-height sampling rate thin-plate spline smoothing on ground truth and prediction.
# D.  add 3 patient as independent test set.

# July 14th, Wednesday, 2021:
# A  uniform output 200x512x200(BxHxW)for all kind of raw image.  --done
# B  if input image has 6 surface, do not extract surfaces.   --done
# C  add pependicular Bscan in z-y  plane as data agumentation, while normal B-scan is in z-x plane.  --done
# D  do not flip OS images.  --done.
# E  output smoothed xml ground truth in orginal image dimension.   --done.


# need python package: simpleitk, scipy, matplotlib, scikit-image

import os
import glob as glob
import SimpleITK as sitk
import json

import matplotlib.pyplot as plt
from utilities import  getSurfacesArray
import sys
sys.path.append("../../..")
from OCTData.OCTDataUtilities import scaleDownMatrix, scaleUpMatrix,BWSurfacesSmooth, saveNumpy2OCTExplorerXML
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


# output image size:
B = 200
H = 512
N = len(extractIndexs)
W = 200  # target image width
C = 1200 # the number of random chosed control points for Thin-Plate-Spline. C is a multiple of 8.
TPSSmoothing = 2.1


# output Dir:
refXMLFile = "/localscratch/Users/hxie1/data/thinRetina/numpy_19cases/refSegXml/PVIP2-4074_Macular_200x200_11-7-2013_8-14-8_OD_sn26558_cube_z_Surfaces_Iowa_Ray.xml"
outputImageDir = "/localscratch/Users/hxie1/data/thinRetina/numpy_19cases/rawGT"
outputNumpyParentDir = "/localscratch/Users/hxie1/data/thinRetina/numpy_19cases"
outputSmoothXmlDir = os.path.join(outputNumpyParentDir, "smoothxml")
outputTrainNumpyDir = os.path.join(outputNumpyParentDir, "training")
outputValidationNumpyDir = os.path.join(outputNumpyParentDir, "validation")
outputTestNumpyDir = os.path.join(outputNumpyParentDir, "test")
outputNoGTTestNumpyDir = os.path.join(outputNumpyParentDir, "noGT_Test1")

if not os.path.exists(outputImageDir):
    os.makedirs(outputImageDir)
if not os.path.exists(outputTrainNumpyDir):
    os.makedirs(outputTrainNumpyDir)
if not os.path.exists(outputValidationNumpyDir):
    os.makedirs(outputValidationNumpyDir)
if not os.path.exists(outputTestNumpyDir):
    os.makedirs(outputTestNumpyDir)
if not os.path.exists(outputSmoothXmlDir):
    os.makedirs(outputSmoothXmlDir)
if not os.path.exists(outputNoGTTestNumpyDir):
    os.makedirs(outputNoGTTestNumpyDir)


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

noGTTestPatientDirList=[
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set3/PVIP2-4100_Macular_512x128_2-4-2010_9-7-27_OS_sn12662_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set3/PVIP2-4119_Macular_200x200_4-7-2015_13-6-21_OD_sn30561_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set3/PVIP2-4124_Macular_200x200_5-13-2010_12-29-8_OD_sn16873_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set3/PVIP2-4105_Macular_512x128_5-19-2010_11-29-4_OD_sn18925_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set3/PVIP2-4122_Macular_200x200_2-27-2012_12-32-59_OS_sn16534_cube_z",
"/localscratch/Users/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set3/PVIP2-4126_Macular_200x200_1-5-2012_9-53-23_OS_sn20578_cube_z",
]

cases= {
    "training": [trainPatientDirList, outputTrainNumpyDir],
    "validation": [validationPatientDirList, outputValidationNumpyDir],
    "test": [testPatientDirList, outputTestNumpyDir],
    "noGTTest1":[noGTTestPatientDirList, outputNoGTTestNumpyDir]
}
for datasetName,[patientDirList, outputNumpyDir] in cases.items():
    outputNumpyImagesPath = os.path.join(outputNumpyDir, f"images.npy")  # for smoothed image
    outputNumpyImagesPathClahe = os.path.join(outputNumpyDir, f"images_clahe.npy")  # for CLAHE image
    outputNumpySurfacesPath = os.path.join(outputNumpyDir, f"surfaces.npy")
    outputPatientIDPath = os.path.join(outputNumpyDir, "patientID.json")

    totalSlices = (B+W)*len(patientDirList)  # B+W is for Bscan and Ascan plane(y-z plane)

    allPatientsImageArray = np.empty((totalSlices , H, W), dtype=float) # for smoothed image
    allPatientsImageArrayClahe = np.empty((totalSlices , H, W), dtype=float) # for CLAHE image
    if "noGTTest" not in datasetName:
        allPatientsSurfaceArray = np.empty((totalSlices, N, W), dtype=float) # the ground truth of data is float
    else:
        allPatientsSurfaceArray = None
    patientIDDict = {}


    print(f"Program is outputing raw_GT images in {outputImageDir} for {datasetName}, please wait ......")
    s = 0 # initial slice index
    for patientDir in patientDirList:
        # get volumePath and surfacesXmlPath
        octVolumeFileList = glob.glob(patientDir + f"/*_OCT_Iowa.mhd")
        assert len(octVolumeFileList) == 1
        octVolumePath = octVolumeFileList[0]
        os.system(f"cp {octVolumePath} {outputSmoothXmlDir}")
        octVolumeRawPath = octVolumePath.replace(".mhd", ".raw")
        os.system(f"cp {octVolumeRawPath} {outputSmoothXmlDir}")


        dirname = os.path.dirname(octVolumePath)
        basename = os.path.basename(octVolumePath)
        basename = basename[0:basename.rfind("_OCT_Iowa.mhd")]
        if "noGTTest" not in datasetName:
            surfacesXmlPath = os.path.join(dirname, basename+f"_Surfaces_Iowa_Ray.xml")
            if not os.path.isfile(surfacesXmlPath):
                surfacesXmlPath = os.path.join(dirname, basename+f"_Surfaces_Iowa.xml")
                if not os.path.isfile(surfacesXmlPath):
                    print("Error: can not find surface xml file")
                    assert False

        else:
            surfacesXmlPath = None

        #  convert Ray's special raw format to standard BxHxW for image, and BxSxW format for surface.
        #  Ray mhd format in BxHxW dimension, but it flip the H and W dimension.
        #  for 200x1024x200 image, and 128x1024x512 in BxHxW direction.
        itkImage = sitk.ReadImage(octVolumePath)
        npImage = sitk.GetArrayFromImage(itkImage).astype(float)  # in BxHxW dimension
        npImage = np.flip(npImage, (1, 2))  # as ray's format filp H and W dimension.
        B0,H0,W0 = npImage.shape  # record original image size.
        B1,H1,W1 = npImage.shape  # 1 indicates image

        if "noGTTest" not in datasetName:
            surfaces = getSurfacesArray(surfacesXmlPath)  # size: SxNxW, where N is number of surfacres.
            B2,N2,W2 = surfaces.shape  # 2 indicates surfaces.
            if N2!= N:
                surfaces = surfaces[:, extractIndexs, :]   #  extract 6 surfaces (0, 1, 3, 5, 6, 10)
                # its surface names: ["ILM", "RNFL-GCL", "IPL-INL", "OPL-HFL", "BMEIS", "OB_RPE"]
                B2, N2, W2 = surfaces.shape  # 2 indicates surfaces.
            assert B1==B2
            assert W1==W2
            assert N2==N
        else:
            surfaces = None


        #  scale down image and surface
        if npImage.shape == (128, 1024, 512):  # scale image to 200x512x200
            # scale down W dimension
            M = scaleDownMatrix(B1, W1, W)
            npImage = np.matmul(npImage, M)  # size: 128x1024x200
            if surfaces is not None:
                surfaces = np.matmul(surfaces, M)  #size: 128xNx200

            # scale up B dimension.
            npImage = np.swapaxes(npImage,axis1=0,axis2=2)  # size: 200x1024x128 in WxHxB
            W,H1,B1 = npImage.shape
            M = scaleUpMatrix(W,B1,B)
            npImage = np.matmul(npImage,M)  # size: WxH1xB
            npImage = np.swapaxes(npImage, axis1=0, axis2=2)  # size: BxH1xW
            if surfaces is not None:
                surfaces = np.swapaxes(surfaces, axis1=0, axis2=2)  # size: 200xNx128 in WxNxB2
                surfaces = np.matmul(surfaces, M)  # size: WxNxB
                surfaces = np.swapaxes(surfaces, axis1=0, axis2=2)  # size: BxNxW

            # scale down H dimension.
            npImage = np.swapaxes(npImage, axis1=1, axis2=2)  # size: 200x200x1024
            B1, W1, H1 = npImage.shape
            M = scaleDownMatrix(B1, H1, H)
            npImage = np.matmul(npImage, M)  # 200x200x512
            npImage = np.swapaxes(npImage, axis1=1, axis2=2)  # size: 200x512x200
            assert (B, H, W == npImage.shape)

            if surfaces is not None:
                surfaces = surfaces * (H / H1)  # 200xNx200
                assert (B, N, W == surfaces.shape)

        elif npImage.shape == (200, 1024, 200):  # scale image to 200x512x200
            # scale down H dimension.
            npImage = np.swapaxes(npImage,axis1=1,axis2=2) # size: 200x200x1024
            B1,W1,H1 = npImage.shape
            M = scaleDownMatrix(B1,H1,H)
            npImage = np.matmul(npImage,M)  # 200x200x512
            npImage = np.swapaxes(npImage,axis1=1,axis2=2) # size: 200x512x200
            assert (B,H,W == npImage.shape)

            if surfaces is not None:
                surfaces = surfaces*(H/H1) # 200xNx200
                assert (B,N,W == surfaces.shape)



        else:
            print(f"Error: npImage size error")
            assert False


        #  Not flip all OS eyes into OD eyes
        #if "_OS_" in basename:
        #    npImage = np.flip(npImage, 2)
        #    surfaces = np.flip(surfaces, 2)

        # Make sure alll surfaces not interleave, especially the top surface of GCIPL (i.e., surface_1) is NOT above ILM (surface_0)
        #for i in range(1, N):
        #    surfaces[:, i, :] = np.where(surfaces[:, i, :] < surfaces[:, i - 1, :], surfaces[:, i - 1, :],
        #                                 surfaces[:, i, :])

        # use BW surface smooth, and guarantee the topological order.
        if surfaces is not None:
            surfaces =BWSurfacesSmooth(surfaces, smoothSurfaceZero=False)  # this is ground truth, no need smoothSurfaceZero again.


        # Average 3 Bscan smoothing.
        smoothedImage = np.zeros_like(npImage,dtype=float)
        for i in range(B): # along B dimension
            i0 = i-1 if i-1>=0 else 0
            i1 = i
            i2 = i+1 if i+1<B else B-1
            smoothedImage[i,] = (npImage[i0, ] +npImage[i1,] +npImage[i2,])/3.0  # intensity in [0,255] in float
        # smoothing in 3 continous A-scan direction.
        tempImage = smoothedImage.copy()
        for i in range(W): # along W dimension
            i0 = i-1 if i-1>=0 else 0
            i1 = i
            i2 = i+1 if i+1<W else W-1
            smoothedImage[i,] = (tempImage[:,:,i0] +tempImage[:,:,i1] +tempImage[:,:,i2])/3.0  # intensity in [0,255] in float

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
        if surfaces is not None:
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
                    interpolator = RBFInterpolator(controlCoordinates, controlValues, neighbors=None, smoothing=TPSSmoothing,
                                               kernel='thin_plate_spline', epsilon=None, degree=None)
                    surfaces[:, i, :] = interpolator(coordinateSurface).reshape(B, W)
                else:
                    # for scipy 1.6.2
                    interpolator = Rbf(controlCoordinates[:,0], controlCoordinates[:,1], controlValues, function='thin_plate',smooth=TPSSmoothing)
                    surfaces[:, i, :] = interpolator(coordinateSurface[:,0], coordinateSurface[:,1]).reshape(B, W)

            # After TPS Interpolation, the surfaces values may exceed the range of [0,H), so it needs clip.
            # for example PVIP2_4045_B128_s127 and s_126 may exceed the low range.
            surfaces = np.clip(surfaces, 0, H-1)

            # output smoothed surface into xml with original size.
            # surfaces of size BxNxW in H height will restore to B0xNxW0 size and in H0 height.
            restoredSurfaces = surfaces.copy()
            if W<W0:
                M = scaleUpMatrix(B, W, W0)
            elif W>W0:
                M = scaleDownMatrix(B, W, W0)
            else:
                M = None
            if M is not None:
                restoredSurfaces = np.matmul(restoredSurfaces, M)  # size: BxNxW0

            if B < B0:
                M = scaleUpMatrix(W0, B, B0)
            elif B > B0:
                M = scaleDownMatrix(W0, B, B0)
            else:
                M = None
            if M is not None:
                restoredSurfaces = np.swapaxes(restoredSurfaces,axis1=0, axis2=2)  # W0xNxB
                restoredSurfaces = np.matmul(restoredSurfaces, M)  # size: W0xNxB0
                restoredSurfaces = np.swapaxes(restoredSurfaces, axis1=0, axis2=2)  # B0xNxW0

            if H != H0:
                restoredSurfaces *= H0/H

            penetrationChar = 'z'  # use z or y to represent penetration direction.
            # physical size of voxel
            voxelSizeUnit = "um"
            penetrationPixels = H0
            # a 6mm x 6mmx x2mm volume.
            if B0==200 and W0==200:
                voxelSizeX = 30.150749
                voxelSizeY =  30.150749
                voxelSizeZ =  1.955034
            elif B0==128 and W0 ==512:
                voxelSizeX = 11.741680
                voxelSizeY = 47.244091
                voxelSizeZ = 1.955034
            else:
                print("Error: maybe image size eror before save xml file.")
                assert False

            saveNumpy2OCTExplorerXML(basename, restoredSurfaces, surfaceNames, outputSmoothXmlDir, refXMLFile,
                                     penetrationChar=penetrationChar, penetrationPixels=penetrationPixels,
                                     voxelSizeUnit=voxelSizeUnit, voxelSizeX=voxelSizeX, voxelSizeY=voxelSizeY,
                                     voxelSizeZ=voxelSizeZ, nameModification="smoothGT")




        #  output  numpy array for all B-scan.
        allPatientsImageArray[s:s+B,:,:] = smoothedImage  # BxHxW
        allPatientsImageArrayClahe[s:s+B,:,:] = claheImage #BxHxW
        if surfaces is not None:
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
            colSubplot = 4 if surfaces is not None else 3
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

            if surfaces is not None:
                subplot4 = plt.subplot(rowSubplot, colSubplot, 4)
                subplot4.imshow(claheImage[i, :, :], cmap='gray')
                for n in range(0, N):
                    subplot4.plot(range(0, W), surfaces[i, n, :], pltColors[n], linewidth=1.2)
                if needLegend:
                    subplot4.legend(surfaceNames, loc='lower left', ncol=2, fontsize='x-small')
                subplot4.axis('off')

            if surfaces is not None:
                curImagePath = os.path.join(outputImageDir, basename+f"_B{B:03d}_s{i:03d}_raw_smoothed_clahe_GT.png")
            else:
                curImagePath = os.path.join(outputImageDir, basename + f"_B{B:03d}_s{i:03d}_raw_smoothed_clahe.png")

            plt.savefig(curImagePath, dpi='figure', bbox_inches='tight', pad_inches=0)
            plt.close()







        #  output  numpy array for all y-z plane
        npImage = np.swapaxes(npImage, axis1=0, axis2=2)  # WxHxB
        smoothedImage = np.swapaxes(smoothedImage, axis1=0, axis2=2)
        claheImage = np.swapaxes(claheImage, axis1=0, axis2=2)
        allPatientsImageArray[s:s + W, :, :] = smoothedImage  # WxHxB
        allPatientsImageArrayClahe[s:s + W, :, :] = claheImage  # WxHxB
        if surfaces is not None:
            surfaces = np.swapaxes(surfaces,axis1=0, axis2=2)
            allPatientsSurfaceArray[s:s + W, :, :] = surfaces  # WxNxB
        for i in range(W):
            # basename: PVIP2-4074_Macular_200x200_11-7-2013_8-14-8_OD_sn26558_cube_z
            patientIDDict[str(s + i)] = basename + f"_W{W:03d}_s{i:03d}"  # e.g. "_W200_s120"
        s += W

        #  out Raw_GT images
        for i in range(W):
            f = plt.figure(frameon=False)
            DPI = 100
            rowSubplot = 1
            colSubplot = 4 if surfaces is not None else 3
            f.set_size_inches(B * colSubplot / float(DPI), H * rowSubplot / float(DPI))

            plt.margins(0)
            plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,
                                hspace=0)  # very important for erasing unnecessary margins.

            subplot1 = plt.subplot(rowSubplot, colSubplot, 1)
            subplot1.imshow(npImage[i, :, :], cmap='gray')
            subplot1.axis('off')

            subplot2 = plt.subplot(rowSubplot, colSubplot, 2)
            subplot2.imshow(smoothedImage[i, :, :], cmap='gray')
            subplot2.axis('off')

            subplot3 = plt.subplot(rowSubplot, colSubplot, 3)
            subplot3.imshow(claheImage[i, :, :], cmap='gray')
            subplot3.axis('off')

            if surfaces is not None:
                subplot4 = plt.subplot(rowSubplot, colSubplot, 4)
                subplot4.imshow(claheImage[i, :, :], cmap='gray')
                for n in range(0, N):
                    subplot4.plot(range(0, W), surfaces[i, n, :], pltColors[n], linewidth=1.2)
                if needLegend:
                    subplot4.legend(surfaceNames, loc='lower left', ncol=2, fontsize='x-small')
                subplot4.axis('off')

            if surfaces is not None:
                curImagePath = os.path.join(outputImageDir, basename + f"_W{W:03d}_s{i:03d}_raw_smoothed_clahe_GT.png")
            else:
                curImagePath = os.path.join(outputImageDir, basename + f"_W{W:03d}_s{i:03d}_raw_smoothed_clahe.png")

            plt.savefig(curImagePath, dpi='figure', bbox_inches='tight', pad_inches=0)
            plt.close()
            

    # after reading all patients, save numpy array
    np.save(outputNumpyImagesPath, allPatientsImageArray)
    np.save(outputNumpyImagesPathClahe, allPatientsImageArrayClahe)
    if allPatientsSurfaceArray is not None:
        np.save(outputNumpySurfacesPath, allPatientsSurfaceArray)
    with open(outputPatientIDPath, 'w') as fp:
        json.dump(patientIDDict, fp)

print(f"===========END of Convert data==============")
