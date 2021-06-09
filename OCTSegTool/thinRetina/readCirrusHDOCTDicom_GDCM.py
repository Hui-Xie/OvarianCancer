# read Cirrus HD-OCT 6000 dicom, version 11.5.1

DicomDir= "/home/hxie1/data/Ophthalmology/3Dicom/Test/Orig_Output_from_Zeiss/01082671/2017-02-13"
outputDir = "/home/hxie1/temp/"

import glob as glob
import gdcm
import os


dicomFileList = glob.glob(DicomDir + f"/*.dcm")
dicomFileList.sort()

for dicomPath in dicomFileList:
    reader = gdcm.Reader()
    reader.SetFileName(dicomPath)
    file = reader.GetFile()
    # Retrieve header information
    fileMetaInformation = file.GetHeader()
    # Retrieve data set
    dataSet = file.GetDataSet()



    dicomData = pydicom.filereader.dcmread(dicomPath)
    patientID = dicomData.PatientID
    visitDate = dicomData.ContentDate
    ODOS = dicomData.Laterality
    seriesDescription = dicomData.SeriesDescription.replace(" ", "")
    outputFilename = f"{patientID}_{visitDate}_{ODOS}_{seriesDescription}.tiff"
    outputPath = os.path.join(outputDir, outputFilename)

    #pixelData = dicomData.PixelData
    pixelData = dicomData.pixel_array
    with Image.open(BytesIO(pixelData)) as im:
        im.save(outputPath)


    print(f"===================")
    break;

