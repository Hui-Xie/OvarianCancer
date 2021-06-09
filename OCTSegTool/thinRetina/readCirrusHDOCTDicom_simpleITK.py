# read Cirrus HD-OCT 6000 dicom, version 11.5.1

DicomDir= "/home/hxie1/data/Ophthalmology/3Dicom/Test/Orig_Output_from_Zeiss/01082671/2017-02-13"
outputDir = "/home/hxie1/temp/"

import glob as glob
import SimpleITK as sitk
from PIL import Image
from io import BytesIO
import os


dicomFileList = glob.glob(DicomDir + f"/*.dcm")
dicomFileList.sort()

'''
reader = sitk.ImageSeriesReader()
#dicom_names = reader.GetGDCMSeriesFileNames(tuple(dicomFileList))
reader.SetFileNames(tuple(dicomFileList))
image = reader.Execute()

'''

image = sitk.ImageRead(dicomFileList[0])
print(f"===================")

