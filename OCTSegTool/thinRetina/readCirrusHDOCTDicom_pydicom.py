# read Cirrus HD-OCT 6000 dicom, version 11.5.1
# it need install openjpeg-2.4.0 and pydicom

import glob as glob
import pydicom
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

'''

def printUsage(argv):
    print("============ Read Dicom Visit directory, extract information =============")
    print("Usage:")
    print(argv[0], "  DicomVisiDir   OutputDir")

def readDicomVisitDir(visitDir, outputDir):
    dicomFileList = glob.glob(visitDir + f"/*.dcm")
    dicomFileList.sort()
    outputFileList = []
    errorFileList = []

    for dicomPath in dicomFileList:
        dicomData = pydicom.filereader.dcmread(dicomPath)
        patientID = dicomData.PatientID
        if hasattr(dicomData, 'ContentDate'):
            visitDate = dicomData.ContentDate
        elif hasattr(dicomData, 'InstanceCreationDate'):
            visitDate = dicomData.InstanceCreationDate
        elif hasattr(dicomData, 'PerformedProcedureStepStartDate'):
            visitDate = dicomData.PerformedProcedureStepStartDate
        else:
            print(f"Error: {patientID} does not has visitDate")
            assert  False
        ODOS = dicomData.Laterality
        modality = dicomData.Modality
        typeTag = dicomData[((0x2201, 0x1000))].value

        if typeTag == "OphthalmicPhotography8BitImage" \
           or  typeTag == "OphthalmicTomographyImage"  \
           or  typeTag == "HfaPerimetryOphthalmicPhotography8BitImage":

            if hasattr(dicomData,'SeriesDescription'):
                seriesDescription = dicomData.SeriesDescription.replace(" ", "")
                outputName = f"{patientID}_{visitDate}_{ODOS}_{modality}_{typeTag}_{seriesDescription}"
            else:
                outputName = f"{patientID}_{visitDate}_{ODOS}_{modality}_{typeTag}"
            outputMhd = outputName + ".mhd"
            outputRaw = outputName + ".raw"
            pixelData = dicomData.pixel_array
            pixelSpacing = (1, 1, 1)
            if hasattr(dicomData,'PixelSpacing'):
                pixelSpacing = dicomData.PixelSpacing

            with open(os.path.join(outputDir,outputMhd), "w") as mhdFile:
                mhdFile.write(f"ObjectType = Image\n")
                mhdFile.write(f"NDims = {pixelData.ndim}\n")
                mhdFile.write(f"BinaryData = True\n")
                mhdFile.write(f"BinaryDataByteOrderMSB = False\n")
                mhdFile.write(f"CompressedData = False\n")

                mhdFile.write(f"ElementSpacing =");
                for x in pixelSpacing:
                    mhdFile.write(f" {x} ")
                mhdFile.write(f"\n")

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
                outputName = f"{patientID}_{visitDate}_{ODOS}_{modality}_{typeTag}_{seriesDescription}_{documentTitle}.pdf"
            else:
                outputName = f"{patientID}_{visitDate}_{ODOS}_{modality}_{typeTag}_{documentTitle}.pdf"
            outputPath = os.path.join(outputDir, outputName)
            pdfFile = open(outputPath,"wb")
            pdfFile.write(pdfData)
            pdfFile.close()
            outputFileList.append(outputName)
        elif typeTag == "HfaOphthalmicVisualFieldStaticPerimetryMeasurements":
            if hasattr(dicomData, 'seriesDescription'):
                seriesDescription = dicomData.SeriesDescription.replace(" ", "")
                outputName = f"{patientID}_{visitDate}_{ODOS}_{modality}_{typeTag}_{seriesDescription}.raw"
            else:
                outputName = f"{patientID}_{visitDate}_{ODOS}_{modality}_{typeTag}.raw"

            if hasattr(dicomData, "pixel_array"):
                 print(f"Warning: {outputName} need process.")
            else:
                errorFileList.append(outputName)
                # Unable to convert the pixel data: one of Pixel Data, Float Pixel Data or Double Float Pixel Data must be present in the dataset
        elif typeTag[-7:] == "RawData":
            if hasattr(dicomData, 'seriesDescription'):
                seriesDescription = dicomData.SeriesDescription.replace(" ", "")
                outputName = f"{patientID}_{visitDate}_{ODOS}_{modality}_{typeTag}_{seriesDescription}.raw"
            else:
                outputName = f"{patientID}_{visitDate}_{ODOS}_{modality}_{typeTag}.raw"

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
    print(f"Below files can not output as program can not read pixel_array")
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