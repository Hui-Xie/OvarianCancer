# utilities functions.

import numpy as np
# import xml.etree.ElementTree as ET
from lxml import etree as ET
import os
import datetime

def extractPaitentID(str):  # for Tongren data
    '''

       :param str: "/home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
       :return: output: 4511_OD_29134
       '''
    stem = str[:str.rfind("_Volume/")]
    patientID = stem[stem.rfind("/") + 1:]
    return patientID

def extractFileName(str): # for Tongren data
    '''

    :param str: "/home/hxie1/data/OCT_Tongren/control/4511_OD_29134_Volume/20110629044120_OCT06.jpg"
    :return: output: 4511_OD_29134_OCT06
    '''
    stem = str[:str.rfind("_Volume/")]
    patientID = stem[stem.rfind("/")+1:]
    OCTIndex = str[str.rfind("_"):-4]
    return patientID+OCTIndex

def loadInputFilesList(filename):
    filesList = []
    with open( filename, "r") as f:
        for line in f:
            filesList.append(line.strip())
    return filesList

def saveInputFilesList(filesList, filename):
    with open( filename, "w") as f:
        for file in filesList:
            f.write(file + "\n")

def getSurfacesArray(segFile):
    """
    Read segmentation result into numpy array from OCTExplorer readable xml file.

    :param segFile in xml format
    :return: array in  [Slices, Surfaces,X] order
    """
    xmlTree = ET.parse(segFile)
    xmlTreeRoot = xmlTree.getroot()

    size = xmlTreeRoot.find('scan_characteristics/size')
    for item in size:
        if item.tag == 'x':
            X =int(item.text)
        elif item.tag == 'y':
            Y = int(item.text)
        elif item.tag == 'z':
            Z = int(item.text)
        else:
            continue

    surface_num = int(xmlTreeRoot.find('surface_num').text)
    surfacesArray = np.zeros((Z, surface_num, X),dtype=float)

    s = -1
    for surface in xmlTreeRoot:
        if surface.tag =='surface':
            s += 1
            z = -1
            for bscan in surface:
                if bscan.tag =='bscan':
                   z +=1
                   x = -1
                   for item in bscan:
                       if item.tag =='y':
                           x +=1
                           surfacesArray[z,s,x] = int(item.text)
    return surfacesArray

def getPatientID_Slice(fileName):
    '''

    :param fileName: e.g. "/home/hxie1/data/OCT_Tongren/control/4162_OD_23992_Volume/20110616012458_OCT10.jpg"
    :return: e.g. '4162_OD_23992, OCT10'
    '''
    splitPath = os.path.split(fileName)
    s = os.path.basename(splitPath[0])
    patientID = s[0:s.rfind("_")]
    s = splitPath[1]
    sliceID = s[s.rfind("_") + 1:s.rfind('.jpg')]
    return patientID, sliceID

def savePatientsPrediction(referFileDir, patientsDict, outputDir, y=496, voxelSizeY=3.87):
    curTime = datetime.datetime.now()
    dateStr = f"{curTime.month:02d}/{curTime.day:02d}/{curTime.year}"
    timeStr = f"{curTime.hour:02d}:{curTime.minute:02d}:{curTime.second:02d}"

    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)

    for patientID in patientsDict.keys():

        #read reference file
        refXMLFile = referFileDir+f"/{patientID}_Volume_Sequence_Surfaces_Iowa.xml"

        # make print pretty
        parser = ET.XMLParser(remove_blank_text=True)
        xmlTree = ET.parse(refXMLFile, parser)
        # old code for ETree
        # xmlTree = ET.parse(refXMLFile)
        xmlTreeRoot = xmlTree.getroot()

        '''
        <modification>
            <date>09/25/2019</date>
            <time>14:40:54</time>
            <modifier>NA</modifier>
            <approval>N</approval>
        </modification>    
        '''
        xmlTreeRoot.find('modification/date').text = dateStr
        xmlTreeRoot.find('modification/time').text = timeStr
        xmlTreeRoot.find('modification/modifier').text = "Hui Xie, Xiaodong Wu"
        ET.SubElement(xmlTreeRoot.find('modification'), 'content', {}).text = "SurfOptNet 1.0"

        xmlTreeRoot.find('scan_characteristics/size/x').text = str(512)

        xmlTreeRoot.find('surface_size/x').text = str(512)
        numSurface = len(patientsDict[patientID].keys())
        xmlTreeRoot.find('surface_num').text= str(numSurface)

        for surface in xmlTreeRoot.findall('surface'):
            xmlTreeRoot.remove(surface)
        for undefinedRegion in xmlTreeRoot.findall('undefined_region'):
            xmlTreeRoot.remove(undefinedRegion)

        for surf in patientsDict[patientID].keys():

            ''' xml format:
            <scan_characteristics>
                <manufacturer>MetaImage</manufacturer>
                <size>
                    <unit>voxel</unit>
                    <x>768</x>
                    <y>496</y>
                    <z>31</z>
                </size>
                <voxel_size>
                    <unit>mm</unit>
                    <x>0.013708</x>
                    <y>0.003870</y>
                    <z>0.292068</z>
                </voxel_size>
                <laterality>NA</laterality>
                <center_type>macula</center_type>
            </scan_characteristics>
            <unit>voxel</unit>
            <surface_size>
                <x>768</x>
                <z>31</z>
            </surface_size>
            <surface_num>11</surface_num>

            <surface>
                <label>10</label>
                <name>ILM (ILM)</name>
                <instance>NA</instance>
                <bscan>
                    <y>133</y>
                    <y>134</y>

            '''
            surfaceElement = ET.SubElement(xmlTreeRoot, 'surface',{})
            ET.SubElement(surfaceElement, 'label',{}).text=str(surf)
            ET.SubElement(surfaceElement, 'name',{}).text = 'ILM(ILM)'
            ET.SubElement(surfaceElement, 'instance',{}).text = 'NA'
            for bscan in patientsDict[patientID][surf].keys():
                bscanElemeent = ET.SubElement(surfaceElement, 'bscan',{})
                for y in patientsDict[patientID][surf][bscan]:
                    ET.SubElement(bscanElemeent, 'y',{}).text = str(y)

        outputXMLFilename =  outputDir + f"/{patientID}_Volume_Sequence_Surfaces_Prediction.xml"
        xmlTree.write(outputXMLFilename, pretty_print=True)
    print(f"{len(patientsDict)} prediction XML surface files are outpted at {outputDir}\n")

def saveNumpy2OCTExplorerXML(patientID, predicition, surfaceNames, outputDir, refXMLFile, y=496, voxelSizeY=3.87):
    curTime = datetime.datetime.now()
    dateStr = f"{curTime.month:02d}/{curTime.day:02d}/{curTime.year}"
    timeStr = f"{curTime.hour:02d}:{curTime.minute:02d}:{curTime.second:02d}"

    # some parameters:
    B, S, W = predicition.shape
    # assert W ==512 and B==31
    assert S == len(surfaceNames)

    # make print pretty
    parser = ET.XMLParser(remove_blank_text=True)
    # read reference file
    xmlTree = ET.parse(refXMLFile, parser)
    xmlTreeRoot = xmlTree.getroot()

    '''
    <modification>
        <date>09/25/2019</date>
        <time>14:40:54</time>
        <modifier>NA</modifier>
        <approval>N</approval>
    </modification>    
    '''
    xmlTreeRoot.find('modification/date').text = dateStr
    xmlTreeRoot.find('modification/time').text = timeStr
    xmlTreeRoot.find('modification/modifier').text = "Hui Xie, Xiaodong Wu"
    ET.SubElement(xmlTreeRoot.find('modification'), 'content', {}).text = "SurfOptNet 1.0"

    xmlTreeRoot.find('scan_characteristics/size/x').text = str(W)
    xmlTreeRoot.find('scan_characteristics/size/y').text = str(y)
    xmlTreeRoot.find('scan_characteristics/size/z').text = str(B)
    xmlTreeRoot.find('scan_characteristics/voxel_size/y').text = str(voxelSizeY/1000)
    xmlTreeRoot.find('surface_size/x').text = str(W)
    xmlTreeRoot.find('surface_size/z').text = str(B)
    xmlTreeRoot.find('surface_num').text = str(S)

    for surface in xmlTreeRoot.findall('surface'):
        xmlTreeRoot.remove(surface)
    for undefinedRegion in xmlTreeRoot.findall('undefined_region'):
        xmlTreeRoot.remove(undefinedRegion)

    for s in range(0,S):

        ''' xml format:
        <scan_characteristics>
            <manufacturer>MetaImage</manufacturer>
            <size>
                <unit>voxel</unit>
                <x>768</x>
                <y>496</y>
                <z>31</z>
            </size>
            <voxel_size>
                <unit>mm</unit>
                <x>0.013708</x>
                <y>0.003870</y>
                <z>0.292068</z>
            </voxel_size>
            <laterality>NA</laterality>
            <center_type>macula</center_type>
        </scan_characteristics>
        <unit>voxel</unit>
        <surface_size>
            <x>768</x>
            <z>31</z>
        </surface_size>
        <surface_num>11</surface_num>

        <surface>
            <label>10</label>
            <name>ILM (ILM)</name>
            <instance>NA</instance>
            <bscan>
                <y>133</y>
                <y>134</y>

        '''
        surfaceElement = ET.SubElement(xmlTreeRoot, 'surface', {})
        ET.SubElement(surfaceElement, 'label', {}).text = str(s)
        ET.SubElement(surfaceElement, 'name', {}).text = surfaceNames[s]
        ET.SubElement(surfaceElement, 'instance', {}).text = 'NA'
        for b in range(B):
            bscanElemeent = ET.SubElement(surfaceElement, 'bscan', {})
            surface = predicition[b,s,:]
            for i in range(W):
                ET.SubElement(bscanElemeent, 'y', {}).text = str(surface[i])

    outputXMLFilename = outputDir + f"/{patientID}_Sequence_Surfaces_Prediction.xml"
    xmlTree.write(outputXMLFilename, pretty_print=True)

def batchPrediciton2OCTExplorerXML(testOutputs, testIDs, numBscan, surfaceNames, outputDir,
                                   refXMLFile="/home/hxie1/data/OCT_Tongren/refXML/1062_OD_9512_Volume_Sequence_Surfaces_Iowa.xml",
                                    y=496, voxelSizeY=3.87, dataInSlice=False):
    B,S,W = testOutputs.shape
    assert B == len(testIDs)
    assert 0 == B%numBscan
    # refXMLFile = "/home/hxie1/data/OCT_Tongren/refXML/1062_OD_9512_Volume_Sequence_Surfaces_Iowa.xml"
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    i=0
    while i<B:
        predicition = testOutputs[i:i+numBscan,:,:]
        dirPath, fileName = os.path.split(testIDs[i])
        for j in range(i+1,i+numBscan):
            dirPath1, fileName1 = os.path.split(testIDs[j])
            if dirPath !=  dirPath1:
                print(f"Error: testID is not continous in {testIDs[j]} against {dirPath}")
                assert False
                return
        if dataInSlice or dirPath=="":
            patientID = fileName[0:fileName.find("_s00.npy")]
        else: # data in volume
            if "/OCT_Tongren/" in dirPath:
                patientID = os.path.basename(dirPath)
            elif "/OCT_JHU/" in dirPath:
                patientID = fileName[0:fileName.rfind("_")]
            else:
                print(f"dirPath: {dirPath}")
                print(f"fileName: {fileName}")
                print(f"Program can not extract volume name correttly")
                assert False

        saveNumpy2OCTExplorerXML(patientID, predicition, surfaceNames, outputDir, refXMLFile, y=y, voxelSizeY=voxelSizeY)
        i += numBscan
