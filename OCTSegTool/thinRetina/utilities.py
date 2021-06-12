# some utilities functions
import numpy as np
# import xml.etree.ElementTree as ET
from lxml import etree as ET

# for ray's surface xml file, which use z axis to express penetration direction.

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
            W =int(item.text)  # Bscan width direction
        elif item.tag == 'y':
            B = int(item.text) # vertical to Bscan, Slices.
        elif item.tag == 'z':
            H = int(item.text) # penetration direction
        else:
            continue

    surface_num = int(xmlTreeRoot.find('surface_num').text)
    surfacesArray = np.zeros((B, surface_num, W))

    n = -1  # surface index
    for surface in xmlTreeRoot:
        if surface.tag =='surface':
            n += 1
            s = -1  # slice index
            for bscan in surface:
                if bscan.tag =='bscan':
                   s +=1
                   w = -1
                   for item in bscan:
                       if item.tag =='z':
                           w +=1
                           surfacesArray[s,n,w] = int(item.text)
    return surfacesArray

def getAllSurfaceNames(segFile):
    """
        Return all surface name defined by OCTExplorer readable xml file.

        :param segFile in xml format
        :return: a list of strings.
        """
    xmlTree = ET.parse(segFile)
    xmlTreeRoot = xmlTree.getroot()

    surface_num = int(xmlTreeRoot.find('surface_num').text)
    print(f"total {surface_num} surfaces.")
    namesList = []

    for surface in xmlTreeRoot:
        if surface.tag == 'surface':
            for name in surface:
                if name.tag == 'name':
                    namesList.append(name.text)

    return namesList

def scaleMatrix(B,W1,W2):
    '''
    return a scale matrix with W1xW2 size with batch B.
    it scale image and surface from HxW1 to HxW2
    :param B:
    :param W1:
    :param W2:
    :return:
    '''
    M = np.zeros((W1, W2))  # scale matrix
    s = W1*1.0/W2 # scale factor
    sr = s # remaining s waiting for allocating along the current column
    sp = 0 # spare space to fill to 1.
    rp = 0 # previous row.
    for c in range(0, W2):
        for r in range(rp,W1):
            if sp != 0:
                M[r,c] = sp
                sr = s- sp
                sp = 0
            elif sr > 1:
                M[r,c] = 1.0
                sr -= 1.0
            else:  #  1>= sr >0
                M[r, c] = sr
                sp = 1.0 - sr
                sr = s
                if sp ==0:
                    rp = r+1
                else:
                    rp = r
                break
    M = M/s  # normalization along each column
    M = np.expand_dims(M,axis=0) # 1xW1xW2
    M = np.repeat(M,B,axis=0)
    return M  # BxW1xW2

def get3PointSmoothMatrix(B,W):
    '''
    return a 3-point Smooth matrix for smoothing ground truth.
    :param B:
    :param W:
    :return:
    '''
    M = np.zeros(W,W)
    M[0,0] = 0.5
    M[1,0] = 0.5
    M[W-2, W-1] = 0.5
    M[W-1, W-1] = 0.5
    for i in range(1, W-1):
        M[i-1,i] = 1.0/3.0
        M[i, i]  = 1.0/3.0
        M[i+1,i]  = 1.0/3.0
    M = np.expand_dims(M, axis=0)  # 1xWxW
    M = np.repeat(M, B, axis=0)  # BxWxW
    return M  # BxWxW
