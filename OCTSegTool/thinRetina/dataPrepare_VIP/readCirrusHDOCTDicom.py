# read Cirrus HD-OCT 6000 dicom, version 11.5.1
# it need install openjpeg-2.4.0 and pydicom

import glob as glob

import numpy as np
import pydicom
from PIL import Image
import os
import sys


'''
Modality and image type:
modality=OP; typeTag=OphthalmicPhotography8BitImage
modality=OPT; typeTag=OphthalmicTomographyImage
modality=OPT; typeTag=CapeCodMacularCubeRawData
modality=OPT; typeTag=CapeCodMacularCubeAnalysisRawData
modality=OPT; typeTag=CapeCodOpticDiscCubeRawData
modality=OPT; typeTag=CapeCodOpticDiscAnalysisRawData
modality=OPT; typeTag=CapeCodOpticDiscCubeRawData
modality=OPT; typeTag=CapeCodOpticDiscAnalysisRawData
modality=OPT; typeTag=EncapsulatedPdf
modality=OPT; typeTag=CapeCodMacularCubeAnalysisRawData
modality=OPT; typeTag=CapeCodGuidedProgressionAnalysisRawData
modality=OT; typeTag=SpecializedEncapsulatedPdf
             typeTag= HfaPerimetryOphthalmicPhotography8BitImage
             typeTag = HfaOphthalmicVisualFieldStaticPerimetryMeasurements
 

===========================================================

mhd format:
ObjectType = Image
NDims = 3
BinaryData = True
BinaryDataByteOrderMSB = False
CompressedData = False
TransformMatrix = 1 0 0 0 1 0 0 0 1
Offset = 0 0 0
CenterOfRotation = 0 0 0
AnatomicalOrientation = RAI
ElementSpacing = 1 1 1
ITK_InputFilterName = NrrdImageIO
ITK_original_direction = 1 0 0 0 1 0 0 0 1
ITK_original_spacing = 1 1 1
NRRD_kinds[0] = domain
NRRD_kinds[1] = domain
NRRD_kinds[2] = domain
NRRD_space = left-posterior-superior
DimSize = 200 1024 200
ElementType = MET_UCHAR
ElementDataFile = 01082671_20200219.raw


# need below package to support pydicom
GDCM - this is the package that supports most compressed formats
Pillow, ideally with jpeg and jpeg2000 plugins
jpeg_ls: CharLS
pylibjpeg, with the -libjpeg, -openjpeg and -rle plugins


The 5-line raster scan is the Cirrus HD-OCT's highest density scan. 
It consists of 4,096 A-scans in each of the five lines. 
The length, angle and spacing between the lines can be adjusted to acquire the best view of the area of interest.



'''

useWHS = True # axial order WidthBscan x HeightBscan x Slice in raw file.

def printUsage(argv):
    print("============ Read Dicom Visit directory, extract information =============")
    print("Usage:")
    print(argv[0], "  DicomVisiDir   OutputDir")

def readDicomVisitDir(visitDir, outputDir):
    dicomFileList = glob.glob(visitDir + f"/*.dcm")
    dicomFileList.sort()
    outputFileList = []
    errorFileList = []
    nFundus = 0
    for dicomPath in dicomFileList:
        dicomData = pydicom.filereader.dcmread(dicomPath)
        patientID = dicomData.PatientID
        if hasattr(dicomData, 'ContentDate'):
            visitDate = dicomData.ContentDate
            visitTime = dicomData.ContentTime.replace(".", "")
        elif hasattr(dicomData, 'InstanceCreationDate'):
            visitDate = dicomData.InstanceCreationDate
            visitTime = dicomData.InstanceCreationTime.replace(".", "")
        elif hasattr(dicomData, 'PerformedProcedureStepStartDate'):
            visitDate = dicomData.PerformedProcedureStepStartDate
            visitTime = dicomData.PerformedProcedureStepStartTime.replace(".", "")
        else:
            print(f"Error: {patientID} does not has visitDate")
            assert  False

        if hasattr(dicomData, "Laterality"):
            ODOS = dicomData.Laterality
        elif hasattr(dicomData, 'ImageLaterality'):
            ODOS = dicomData.ImageLaterality
        else:
            ODOS = "ODOS"

        #  (0x2201, 0x1000) in dicomData judge key exist or not


        modality = dicomData.Modality
        try:
            typeTag = dicomData[(0x2201, 0x1000)].value
        except:
            sopClassUID = dicomData[(0x0008, 0x0016)].value
            if sopClassUID == "1.2.840.10008.5.1.4.1.1.104.1":
                typeTag = "EncapsulatedPdf"
            elif sopClassUID == "1.2.840.10008.5.1.4.1.1.7":
                typeTag = "FundusImage"

        if typeTag == "OphthalmicPhotography8BitImage" \
           or  typeTag == "OphthalmicTomographyImage"  \
           or  typeTag == "HfaPerimetryOphthalmicPhotography8BitImage"\
           or  typeTag == "FundusImage":

            if dicomData.NumberOfFrames == 1:
                if dicomData.SamplesPerPixel == 1: # SLO image
                    typeTag = "SLO"
                    outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}.png"
                    pixelData = dicomData.pixel_array
                    Image.fromarray(pixelData).save(os.path.join(outputDir, outputName))
                    outputFileList.append(outputName)

                elif dicomData.SamplesPerPixel == 3: # color fundus image
                    typeTag = "Fundus"
                    # Speciral color interpretation:
                    # (0028, 0004) Photometric Interpretation          CS: 'YBR_FULL_422'
                    outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}_{nFundus:02d}.png"
                    nFundus +=1
                    pixelData = dicomData.pixel_array
                    pixelData = pydicom.pixel_data_handlers.util.convert_color_space(pixelData, dicomData[(0x0028, 0x0004)].value, "RGB")
                    Image.fromarray(pixelData).save(os.path.join(outputDir, outputName))
                    outputFileList.append(outputName)
                else:
                    assert False
                    print(f"Error something wrong with frame=1 ==============")

            else: # OCT volume
                if hasattr(dicomData,'SeriesDescription'):
                    seriesDescription = dicomData.SeriesDescription.replace(" ", "")
                    outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}_{seriesDescription}"
                else:
                    outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}"
                outputMhd = outputName + ".mhd"
                outputRaw = outputName + ".raw"
                pixelData = dicomData.pixel_array # default axial order: Slice x HeightBscan x WidthBscan or SHW order
                if useWHS:
                    pixelData = np.swapaxes(pixelData,0,2)

                if hasattr(dicomData,'PixelSpacing'):
                    pixelSpacing = dicomData.PixelSpacing
                elif hasattr(hasattr(dicomData,"AcrossScanSpatialResolution") and hasattr(dicomData,"DepthSpatialResolution") and dicomData,"AlongScanSpatialResolution") :
                    if useWHS:
                        pixelSpacing = (dicomData.AlongScanSpatialResolution/1000.0, dicomData.DepthSpatialResolution/1000.0, dicomData.AcrossScanSpatialResolution/1000.0)
                    else: # SHW
                        pixelSpacing = (dicomData.AcrossScanSpatialResolution/1000.0, dicomData.DepthSpatialResolution/1000.0, dicomData.AlongScanSpatialResolution/1000.0)
                    # dicom pixel array dimension: Slice x Height x Width or Frames x Rows x Columns
                    # the unit of dicom resolution is microns, needing to convert microns to mm by 1/1000.
                else:
                    pixelSpacing = (1, 1, 1)

                with open(os.path.join(outputDir,outputMhd), "w") as mhdFile:
                    mhdFile.write(f"ObjectType = Image\n")
                    mhdFile.write(f"NDims = {pixelData.ndim}\n")
                    mhdFile.write(f"BinaryData = True\n")
                    mhdFile.write(f"BinaryDataByteOrderMSB = False\n")
                    mhdFile.write(f"CompressedData = False\n")
                    mhdFile.write(f"TransformMatrix = 1 0 0 0 1 0 0 0 1\n")
                    mhdFile.write(f"Offset = 0 0 0\n")
                    mhdFile.write(f"CenterOfRotation = 0 0 0\n")
                    mhdFile.write(f"AnatomicalOrientation = RAI\n")
                    mhdFile.write(f"ElementSpacing =")
                    for x in pixelSpacing:
                        mhdFile.write(f" {x} ")
                    mhdFile.write(f"\n")
                    mhdFile.write(f"ITK_InputFilterName = NrrdImageIO\n")

                    mhdFile.write(f"ITK_original_direction = 1 0 0 0 1 0 0 0 1\n")
                    mhdFile.write(f"ITK_original_spacing = 1 1 1\n")
                    mhdFile.write(f"NRRD_kinds[0] = domain\n")
                    mhdFile.write(f"NRRD_kinds[1] = domain\n")
                    mhdFile.write(f"NRRD_kinds[2] = domain\n")
                    if useWHS:
                        mhdFile.write(f"NRRD_space = left-posterior-superior\n")
                    else:
                        mhdFile.write(f"NRRD_space = inferior-posterior-left\n") # for slice x Height x Width dimension.
                    mhdFile.write(f"DimSize = ")
                    for x in pixelData.shape:
                        mhdFile.write(f" {x} ")
                    mhdFile.write(f"\n")

                    mhdFile.write(f"ElementType = MET_UCHAR\n")
                    mhdFile.write(f"ElementDataFile = {outputRaw}\n")
                pixelData.astype('uint8').tofile(os.path.join(outputDir, outputRaw))
                outputFileList.append(outputMhd)
        elif typeTag[-3:] == "Pdf":
            documentTitle = dicomData.DocumentTitle.replace(" ", "")
            pdfData = dicomData.EncapsulatedDocument
            if hasattr(dicomData, 'seriesDescription'):
                seriesDescription = dicomData.SeriesDescription.replace(" ", "")
                outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}_{seriesDescription}_{documentTitle}.pdf"
            else:
                outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}_{documentTitle}.pdf"
            outputPath = os.path.join(outputDir, outputName)
            pdfFile = open(outputPath,"wb")
            pdfFile.write(pdfData)
            pdfFile.close()
            outputFileList.append(outputName)
        elif typeTag == "HfaOphthalmicVisualFieldStaticPerimetryMeasurements":
            if hasattr(dicomData, 'seriesDescription'):
                seriesDescription = dicomData.SeriesDescription.replace(" ", "")
                outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}_{seriesDescription}.raw"
            else:
                outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}.raw"

            if hasattr(dicomData, "pixel_array"):
                 print(f"Warning: {outputName} need process.")
            else:
                errorFileList.append(outputName)
                # Unable to convert the pixel data: one of Pixel Data, Float Pixel Data or Double Float Pixel Data must be present in the dataset
        elif typeTag[-7:] == "RawData":
            if hasattr(dicomData, 'seriesDescription'):
                seriesDescription = dicomData.SeriesDescription.replace(" ", "")
                outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}_{seriesDescription}.raw"
            else:
                outputName = f"ID{patientID}_D{visitDate}_T{visitTime}_{ODOS}_{modality}_{typeTag}.raw"

            if hasattr(dicomData,"pixel_array"):
                print(f"Warning: {outputName} need process.")
            else:
                errorFileList.append(outputName)
                # Unable to convert the pixel data: one of Pixel Data, Float Pixel Data or Double Float Pixel Data must be present in the dataset
        else:
            print(f"Error: typeTage= {typeTag}")

    # out put summary information:
    print("==================================================")
    print(f"Output below files at {outputDir}:")
    for file in outputFileList:
        print(file)
    print("==================================================")
    print(f"Below files can not output as program can not read pixel_array:")
    for file in errorFileList:
        print(file)
    print("==================================================")


def main():
    if len(sys.argv) != 3:
        print("Error: input parameters error.")
        printUsage(sys.argv)
        return -1
    readDicomVisitDir(sys.argv[1], sys.argv[2])
    print("=================END=========================")

if __name__ == "__main__":
    main()