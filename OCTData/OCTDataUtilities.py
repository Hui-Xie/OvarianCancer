import torch
import datetime
import os
from lxml import etree as ET


def computeErrorStdMuOverPatientDimMean(predicitons, gts, slicesPerPatient=31, hPixelSize=3.870, goodBScansInGtOrder=None):
    '''
    Compute error standard deviation and mean along different dimension.

    First convert absError on patient dimension


    :param predicitons: in (BatchSize, NumSurface, W) dimension, in strictly patient order.
    :param gts: in (BatchSize, NumSurface, W) dimension
    :param hPixelSize: in micrometer
    :param goodBScansInGtOrder:
    :return: muSurface: (NumSurface) dimension, mean for each surface
             stdSurface: (NumSurface) dimension
             muPatient: (NumPatient) dimension, mean for each patient
             stdPatient: (NumPatient) dimension
             mu: a scalar, mean over all surfaces and all batchSize
             std: a scalar
    '''
    device = predicitons.device
    B,N, W = predicitons.shape # where N is numSurface
    absError = torch.abs(predicitons-gts)

    if goodBScansInGtOrder is None:
        P = B // slicesPerPatient
        absErrorPatient = torch.zeros((P,N), device=device)
        for p in range(P):
            absErrorPatient[p,:] = torch.mean(absError[p * slicesPerPatient:(p + 1) * slicesPerPatient, ], dim=(0,2))*hPixelSize
    else:
        P = len(goodBScansInGtOrder)
        absErrorPatient = torch.zeros((P, N), device=device)
        for p in range(P):
            absErrorPatient[p,:] = torch.mean(absError[p * slicesPerPatient+goodBScansInGtOrder[p][0]:p * slicesPerPatient+goodBScansInGtOrder[p][1], ], dim=(0,2))*hPixelSize

    stdSurface, muSurface = torch.std_mean(absErrorPatient, dim=0)
    # size of stdSurface, muSurface: [N]
    std, mu = torch.std_mean(absErrorPatient)
    return stdSurface, muSurface, std,mu

def saveNumpy2OCTExplorerXML(patientID, predicition, surfaceNames, outputDir, refXMLFile,
                             penetrationChar, penetrationPixels,
                             voxelSizeUnit, voxelSizeX, voxelSizeY, voxelSizeZ):
    curTime = datetime.datetime.now()
    dateStr = f"{curTime.month:02d}/{curTime.day:02d}/{curTime.year}"
    timeStr = f"{curTime.hour:02d}:{curTime.minute:02d}:{curTime.second:02d}"

    # some parameters:
    B, S, W = predicition.shape
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
    ET.SubElement(xmlTreeRoot.find('modification'), 'content', {}).text = "SurfaceSegNet"

    xmlTreeRoot.find('scan_characteristics/size/x').text = str(W)
    if penetrationChar=='y':
        xmlTreeRoot.find('scan_characteristics/size/y').text = str(penetrationPixels)
        xmlTreeRoot.find('scan_characteristics/size/z').text = str(B)
    else:
        xmlTreeRoot.find('scan_characteristics/size/y').text = str(B)
        xmlTreeRoot.find('scan_characteristics/size/z').text = str(penetrationPixels)

    xmlTreeRoot.find('scan_characteristics/voxel_size/unit').text = voxelSizeUnit
    xmlTreeRoot.find('scan_characteristics/voxel_size/x').text = str(voxelSizeX)
    xmlTreeRoot.find('scan_characteristics/voxel_size/y').text = str(voxelSizeY)
    xmlTreeRoot.find('scan_characteristics/voxel_size/z').text = str(voxelSizeZ)

    xmlTreeRoot.find('surface_size/x').text = str(W)
    if penetrationChar == 'y':
        xmlTreeRoot.find('surface_size/z').text = str(B)
    else:
        xmlTreeRoot.find('surface_size/y').text = str(B)


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
                ET.SubElement(bscanElemeent, penetrationChar, {}).text = str(surface[i])

    outputXMLFilename = outputDir + f"/{patientID}_Sequence_Surfaces_Prediction.xml"
    xmlTree.write(outputXMLFilename, pretty_print=True)

def batchPrediciton2OCTExplorerXML(testOutputs, volumeIDs, volumeBscanStartIndexList, surfaceNames, outputDir,
                                   refXMLFile="/home/hxie1/data/OCT_Tongren/refXML/1062_OD_9512_Volume_Sequence_Surfaces_Iowa.xml",
                                   penetrationChar='y', penetrationPixels=496, voxelSizeUnit='um', voxelSizeX=13.708, voxelSizeY=3.870, voxelSizeZ=292.068):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    N = len(volumeIDs)
    assert N == len(volumeBscanStartIndexList)
    for i in range(N):
        if i != N-1:
            predicition = testOutputs[volumeBscanStartIndexList[i]:volumeBscanStartIndexList[i+1], :, :]  # prediction volume
        else:
            predicition = testOutputs[volumeBscanStartIndexList[i]:, :, :]  # prediction volume
        saveNumpy2OCTExplorerXML(volumeIDs[i], predicition, surfaceNames, outputDir, refXMLFile,
                                 penetrationChar=penetrationChar, penetrationPixels=penetrationPixels,
                                 voxelSizeUnit=voxelSizeUnit, voxelSizeX=voxelSizeX, voxelSizeY=voxelSizeY, voxelSizeZ=voxelSizeZ)

