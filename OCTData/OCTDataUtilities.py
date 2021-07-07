import torch
import datetime
import os
from lxml import etree as ET
import numpy as np
from scipy import ndimage

def computeErrorStdMuOverPatientDimMean(predicitons, gts, slicesPerPatient=31, hPixelSize=3.870, goodBScansInGtOrder=None):
    '''

    MASD(mean absolute surface distance error, $\mu m$),
    this is for uniform Bscan number for all volumes.

    Compute error standard deviation and mean along different dimension.

    First convert absError on patient dimension

    :param predicitons: in (BatchSize, NumSurface, W) dimension, in strictly patient order.
    :param gts: in (BatchSize, NumSurface, W) dimension
    :param hPixelSize: in micrometer
    :param goodBScansInGtOrder:
    :return: muSurface: (NumSurface) dimension, mean for each surface
             stdSurface: (NumSurface) dimension
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

def computeMASDError(predicitons, gts, volumeBscanStartIndexList, hPixelSize=3.870):
    '''
    Tensor version.
    MASD(mean absolute surface distance error, $\mu m$),
    support different Bscan numbers for different volumes.

    Compute error standard deviation and mean along different dimension.

    First convert absError on patient dimension

    :param predicitons: in (BatchSize, NumSurface, W) dimension, in strictly patient order.
    :param gts: in (BatchSize, NumSurface, W) dimension
    :param volumeBscanStartIndexList
    :param hPixelSize: in micrometer

    :return: muSurface: (NumSurface) dimension, mean for each surface
             stdSurface: (NumSurface) dimension
             mu: a scalar, mean over all surfaces and all batchSize
             std: a scalar
    '''
    device = predicitons.device
    B,N, W = predicitons.shape # where N is numSurface
    absError = torch.abs(predicitons-gts)  # size: B,N,W

    P = len(volumeBscanStartIndexList)
    absErrorPatient = torch.zeros((P, N), dtype=torch.float,device=device)
    for p in range(P):
        if p != P-1:
            absErrorPatient[p,:] = torch.mean(absError[volumeBscanStartIndexList[p]:volumeBscanStartIndexList[p+1], ], dim=(0,2))*hPixelSize
        else:
            absErrorPatient[p, :] = torch.mean(absError[volumeBscanStartIndexList[p]:, ], dim=(0, 2)) * hPixelSize

    stdSurface, muSurface = torch.std_mean(absErrorPatient, dim=0)
    # size of stdSurface, muSurface: [N]
    std, mu = torch.std_mean(absErrorPatient)
    return stdSurface, muSurface, std,mu

def computeMASDError_numpy(predicitons, gts, volumeBscanStartIndexList, hPixelSize=3.870):
    '''
    Numpy version.
    MASD(mean absolute surface distance error, $\mu m$),
    support different Bscan numbers for different volumes.

    Compute error standard deviation and mean along different dimension.

    First convert absError on patient dimension

    :param predicitons: in (BatchSize, NumSurface, W) dimension, in strictly patient order.
    :param gts: in (BatchSize, NumSurface, W) dimension
    :param volumeBscanStartIndexList
    :param hPixelSize: in micrometer
    :return: muSurface: (NumSurface) dimension, mean for each surface
             stdSurface: (NumSurface) dimension
             mu: a scalar, mean over all surfaces and all batchSize
             std: a scalar
    '''
    B,N, W = predicitons.shape # where N is numSurface
    absError = np.absolute(predicitons-gts)  # size: B,N,W

    P = len(volumeBscanStartIndexList)
    absErrorPatient = np.zeros((P, N), dtype=float)
    for p in range(P):
        if p != P-1:
            absErrorPatient[p,:] = np.mean(absError[volumeBscanStartIndexList[p]:volumeBscanStartIndexList[p+1], ], axis=(0,2))*hPixelSize
        else:
            absErrorPatient[p, :] = np.mean(absError[volumeBscanStartIndexList[p]:, ], axis=(0, 2)) * hPixelSize

    stdSurface= np.std(absErrorPatient, axis=0)
    muSurface = np.mean(absErrorPatient, axis=0)
    # size of stdSurface, muSurface: [N]
    std = np.std(absErrorPatient)
    mu  = np.mean(absErrorPatient)
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
                                   penetrationChar='y', penetrationPixels=496, voxelSizeUnit='um', voxelSizeX=13.708, voxelSizeY=3.870, voxelSizeZ=292.068, OSFlipBack=False):
    if not os.path.isdir(outputDir):
        os.makedirs(outputDir)
    N = len(volumeIDs)
    assert N == len(volumeBscanStartIndexList)
    for i in range(N):
        if i != N-1:
            prediction = testOutputs[volumeBscanStartIndexList[i]:volumeBscanStartIndexList[i+1], :, :]  # prediction volume
        else:
            prediction = testOutputs[volumeBscanStartIndexList[i]:, :, :]  # prediction volume

        if ("_OS_" in volumeIDs[i]) and OSFlipBack:
            prediction = np.flip(prediction, 2)

        saveNumpy2OCTExplorerXML(volumeIDs[i], prediction, surfaceNames, outputDir, refXMLFile,
                                 penetrationChar=penetrationChar, penetrationPixels=penetrationPixels,
                                 voxelSizeUnit=voxelSizeUnit, voxelSizeX=voxelSizeX, voxelSizeY=voxelSizeY, voxelSizeZ=voxelSizeZ)

def outputNumpyImagesSegs(images, segs, volumeIDs, volumeBscanStartIndexList, outputDir):
    '''

    :param images:  in numpy format
    :param segs: in numpy format.
    :param volumeIDs:  a list
    :param volumeBscanStartIndexList:
    :param outputDir:
    :return:
    '''
    B,H,W = images.shape
    B1,N,W1 = segs.shape
    assert B==B1 and W ==W1
    numVolumes = len(volumeIDs)
    assert numVolumes == len(volumeBscanStartIndexList)
    for i in range(numVolumes):
        if i != numVolumes-1:
            image = images[volumeBscanStartIndexList[i]:volumeBscanStartIndexList[i+1],]
            seg = segs[volumeBscanStartIndexList[i]:volumeBscanStartIndexList[i+1],]
        else:
            image = images[volumeBscanStartIndexList[i]:, ]
            seg = segs[volumeBscanStartIndexList[i]:, ]

        np.save(os.path.join(outputDir,f"{volumeIDs[i]}_volume.npy"), image)
        np.save(os.path.join(outputDir,f"{volumeIDs[i]}_segmentation.npy"), seg)

def smoothSurfacesWithPrecision(mu, precision, windSize= 5):
    '''
    For each surface,  implement a mu_s * M in dimension (Bx1xW)  *  (BxWxW), where M is a precision-weighed smooth matrix
    with slide window size windSize. and then assembly all smoothed surface.
    Putting S out outer loop is to utilize matrix bmm.
    :param mu:
    :param sigma2:
    :param windSize:
    :return:
    '''
    B,S,W = precision.shape
    assert mu.shape == (B,S,W)
    device = precision.device

    assert windSize%2==1
    h = windSize//2  # half of the slide window size

    smoothMu = torch.zeros_like(mu)
    for s in range(S):
        mus = mu[:,s, :].unsqueeze(dim=1)  # s indicate specific surface, size: Bx1xW
        P  = precision[:,s, :] # size: BxW

        M = torch.zeros((B,W, W), dtype=torch.float32, device=device)  # smooth matrix
        for c in range(0,W):
            top = c-h
            bottom = c+h+1 # columen low boundary outside
            if top < 0:
                offset = -top
                top +=offset
                bottom -=offset
            if bottom > W:
                offset = bottom-W
                bottom -=offset
                top  +=offset
            colSumP = torch.sum(P[:,top:bottom], dim=-1,keepdim=True) # size: Bx1
            colSump = colSumP.expand(B,bottom-top)  # size: Bx(bottom-top)
            M[:,top:bottom,c] = P[:,top:bottom]/colSumP  # size: Bx(bottom-top)

        smoothMu[:,s,:] = torch.bmm(mus, M).squeeze(dim=1)     # size: Bx1xW -> BxW

    return smoothMu # BxSxW

def adjustSurfacesUsingPrecision(S, P):
    '''
    Choose surface value with a bigger precision when surfaces conflict.
    :param S: surface tensor in BxNxW
    :param P: Precision tensor in BxNxW
    :return: adjusted Su
    '''
    B,N,W = S.shape
    errorPoints = torch.nonzero(S[:, 0:-1, :] > S[:, 1:, :], as_tuple=False)
    for i in range(errorPoints.shape[0]):
        b = errorPoints[i, 0].item()
        s = errorPoints[i, 1].item()
        w = errorPoints[i, 2].item()
        # at this point: S[b,s,w] > S[b,s+1,w]
        # first use majority rule, an invalidate surface crossing 2+ surfaces is an error
        if s+2<N and S[b,s,w] >= S[b,s+2,w]:
            S[b, s, w] = S[b, s + 1, w]
        if s-1>=0 and S[b,s+1,w] <= S[b,s-1,w]:
            temp = S[b, s, w]
            S[b, s, w] = 0.5*( S[b, s + 1, w] + S[b, s, w])
            S[b, s + 1, w] = S[b, s, w] + 0.5*(temp-S[b, s, w])
        # choose the surface value with bigger precision when surfaces conflict
        if S[b,s,w] > S[b,s+1,w]:
            if P[b,s,w] > P[b,s+1,w]:
                S[b,s+1,w] = S[b,s,w]  #ReLU
            else:
                S[b,s,w] = S[b,s+1,w]

    return S


def medianFilterSmoothing(input, winSize=21):
    '''
    apply 1D median filter along W direction at the outlier points  only.
    :param input: in size of BxSxW
    :param winSize: a int scalar, a odd number
    :return:
    '''
    B,S,W = input.shape
    ndim = input.ndim
    mInput, _ = torch.median(input, dim=-1, keepdim=True) # size: BxSx1
    mInput = mInput.expand_as(input) # size: BxSxW

    h = winSize//2 # half winSize
    output = input.clone()

    #scaled median absolute deviation (MAD)
    # ref: https://www.mathworks.com/help/matlab/ref/isoutlier.html#bvolfgk
    c = 1.4826
    MAD, _ = torch.median((input-mInput).abs(), dim=-1,keepdim=True) # size: BxSx1
    MAD = (c*MAD).expand_as(input) # size: BxSxW

    # an outlier is a value that is more than three scaled median absolute deviations (MAD) away from the median.
    factor = 3.0 # instead of 3 as matlab
    outlierIndexes = torch.nonzero((input-mInput).abs() >= factor*MAD, as_tuple=False)
    N,dims = outlierIndexes.shape
    assert dims ==ndim
    for i in range(N):
        b = int(outlierIndexes[i,0])
        s = int(outlierIndexes[i,1])
        w = int(outlierIndexes[i,2])
        low = w-h
        high = w+h+1 # outside high boundary with 1
        if low<0:
            offset = -low
            low +=offset
            high +=offset
        if high>W:
            offset = high-W
            high -= offset
            low  -= offset
        output[b,s,w] = torch.median(input[b,s,low:high])

    return output

def BWSurfacesSmooth(surfaces):
    '''
     Smooth each enface surface if there are topological order violations, with inplace median smoothing.
    :param surfaces: in size of BxSxW for a volume in numpy array.
    :param winSize: a int scalar, an odd number
    :return:
    '''
    B,S,W = surfaces.shape
    ndim = surfaces.ndim

    MaxInteration = S*S # s+1 change may affect its next neighbor changes.

    # median filter each surface to erase noises generated by image artifacts.
    surfaces = ndimage.median_filter(surfaces, size=(5,1,5))  # 5x5 is better than 3x3 and 7x7.
    # surface 0 add extra median_filter as surface 0 is smooth and easy to recognize
    surfaces[:,0,:] = ndimage.median_filter(surfaces[:,0,:], size=(5, 5))  # 5x5 is better than 3x3 and 7x7.

    # the outlier are s_{i-1} > s_{i}
    outlierIndexes = np.transpose(np.nonzero(surfaces[:,0:-1,:] > surfaces[:,1:,:])) # as_tuple=False
    N,dims = outlierIndexes.shape
    outputSurfaces = surfaces.copy()
    assert dims ==ndim
    nIterations = 0
    while (N > 0):
        nIterations +=1
        if nIterations > MaxInteration:
            print(f"BWSurfaceSmooth exit exceeding the MaxInteration={MaxInteration}")
            break

        for i in range(N):
            b,s,w = outlierIndexes[i,]  # s is current surface.
            # surface coordinates 5x5 neighbor.
            blow = b-2 if b-2 >=0 else 0
            bhigh = b+3 if b+3 <=B else B
            wlow = w-2 if w-2>=0  else 0
            whigh = w+3 if w+3<=W else W
            surface0 = np.median(surfaces[blow:bhigh, s, wlow:whigh])
            surface1 = np.median(surfaces[blow:bhigh, s+1, wlow:whigh])
            if surface0 > surface1:
                # the surface with smaller std is more reliable.
                std0 = np.std(surfaces[blow:bhigh, s, wlow:whigh])
                std1 = np.std(surfaces[blow:bhigh, s+1, wlow:whigh])
                # uniform at the surface with smaller std.
                if std0 < std1:
                    surface1 = surface0
                else:
                    surface0 = surface1
            outputSurfaces[b, s, w] = surface0
            outputSurfaces[b, s + 1, w] = surface1

        # modfity the relation of s and s+1, then s+1 and s+2 relation may be affected. So it needs to check again.
        surfaces = outputSurfaces.copy()
        outlierIndexes = np.transpose(np.nonzero(surfaces[:, 0:-1, :] > surfaces[:, 1:, :]))  # as_tuple=False
        N, dims = outlierIndexes.shape
    return outputSurfaces

def scaleUpMatrix(B, W1, W2):
    '''
    return a scale matrix with W1xW2 size with batch size B.
    it scale up surface and image  from BxNxW1 to BxNxW2, where W1 < W2
    :param B:
    :param W1:
    :param W2:
    :return:
    '''
    M = np.zeros((W1, W2),dtype=float)  # scale matrix
    assert W2 > W1
    s = W2*1.0/W1 # scale factor
    sr = s # remaining s waiting for allocating along the current row
    sp = 0 # spare space to fill to 1.
    cp = 0 # previous column.
    # sum(eachRow in M) = s, and sum(each column in M) = 1
    for r in range(0, W1):
        for c in range(cp,W2):
            if sp != 0:
                M[r,c] = sp
                sr = s- sp
                sp = 0
            elif sr > 1.0:
                M[r,c] = 1.0
                sr -= 1.0
            else: #  1>= sr >0
                M[r, c] = sr
                sp = 1.0 - sr
                sr = s
                if sp == 0:
                    cp = c + 1
                else:
                    cp = c
                break

    M = np.expand_dims(M,axis=0) # 1xW1xW2
    M = np.repeat(M,B,axis=0)
    return M  # BxW1xW2




