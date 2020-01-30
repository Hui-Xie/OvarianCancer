# utilities functions.

import numpy as np
# import xml.etree.ElementTree as ET
from lxml import etree as ET
import os
import datetime

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
    surfacesArray = np.zeros((Z, surface_num, X))

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

def savePatientsPrediction(referFileDir, patientsDict, outputDir):
    curTime = datetime.datetime.now()
    dateStr = f"{curTime.month:02d}/{curTime.day:02d}/{curTime.year}"
    timeStr = f"{curTime.hour:02d}:{curTime.minute:02d}:{curTime.second:02d}"

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
        xmlTreeRoot.find('modification/modifier').text = "Xiaodong Wu, Leixin Zhou, Hui Xie"
        ET.SubElement(xmlTreeRoot.find('modification'), 'content', {}).text = "Model-Based Deep Learning for Globally Optimal Surface Segmentation"

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

