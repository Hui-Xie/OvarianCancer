# generate 9 sector thickness map for all 9-layer thickness map

inputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thicknessEnfaceMap_9x31x25"
outputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/thickness9Sector_9x9"

H = 31  # odd number
W = 25  # odd number
inputFileSuffix = "_Volume_thickness_enface.npy"
outputFileSuffix = "_thickness9sector_9x9.npy"
nLayers = 9  # number of layers
nSectors = 9 # 9 sector polar map

import numpy as np
import cmath
import math
import glob
import os

Pi = math.pi

def generatePolarMap(H, W):
    '''

    :param H: input image Height
    :param W: input image Weight.
    :return: 2xHxW array polarMap
             polarMask[0]: is the modulus, radius >=0
             polarMask[1]: is the phase, angle in range(-Pi, Pi]

    '''
    # generate rectangle coordinates map
    recMap = np.ones((2,H,W), dtype=np.int)*(-1) # recMap[0] is x coordinate, and recMap[1] is y coordinate
    recMap[0,] = np.repeat(np.arange(-(W//2), W//2+1).reshape(1, W), H, axis=0)
    recMap[1,] = np.repeat(np.arange(H // 2, -(H // 2) - 1, -1).reshape(H, 1), W, axis=1)  # Y coordinate upward.
    # convert to polar coordinates
    polarMap = np.ones((2,H,W), dtype=np.float)*(-1)
    for i in range(H):
        for j in range(W):
            polarMap[0,i,j], polarMap[1,i,j] = cmath.polar(complex(recMap[0,i,j], recMap[1,i,j])) # phase is in range(-Pi,Pi]
    return polarMap

def get9SectorsMask(polarMap, OS_OD):
    '''

    :param polarMap:
    :param OS_OD:  "OD"(right) or "OS" (left)
    :return: sectorsMask with [0,8] to mask different sectors, and -1 indicate non-selection sector.
    '''
    C,H,W = polarMap.shape
    assert C==2
    stepR = min(H,W)//6
    sectorsmask = np.ones((H,W), dtype=np.int)*(-1)
    for i in range(H):
        for j in range(W):
            p = polarMap[0,i,j]
            theta = polarMap[1,i,j]
            if p <= stepR:
                sectorsmask[i,j] = 0
            elif -Pi/4 <= theta < Pi/4:  # 1,3,5,7
                if stepR < p <= 2*stepR:
                    if OS_OD =="OD":
                        sectorsmask[i, j] = 1
                    else:
                        sectorsmask[i, j] = 3
                if 2*stepR < p <= 3*stepR:
                    if OS_OD =="OD":
                        sectorsmask[i, j] = 5
                    else:
                        sectorsmask[i, j] = 7

            elif Pi/4 <= theta < Pi*3/4: # 2,6
                if stepR < p <= 2 * stepR:
                    sectorsmask[i, j] = 2
                if 2 * stepR < p <= 3 * stepR:
                    sectorsmask[i, j] = 6

            elif (Pi*3/4 <= theta)  or (theta < -Pi*3/4): # 1,3,5,7
                if stepR < p <= 2 * stepR:
                    if OS_OD == "OD":
                        sectorsmask[i, j] = 3
                    else:
                        sectorsmask[i, j] = 1
                if 2 * stepR < p <= 3 * stepR:
                    if OS_OD == "OD":
                        sectorsmask[i, j] = 7
                    else:
                        sectorsmask[i, j] = 5
            elif -Pi*3/4 <= theta < -Pi/4: #4, 8
                if stepR < p <= 2 * stepR:
                    sectorsmask[i, j] = 4
                if 2 * stepR < p <= 3 * stepR:
                    sectorsmask[i, j] = 8
            else:
                print(f"Error in phase from polarmap")
                assert False
    return sectorsmask

def printSectorMask(sectorMask):
    print("========sectorMask=========")
    H,W = sectorMask.shape
    for i in range(H):
        for j in range(W):
            if sectorMask[i,j]==-1:
                print(" ", end="")
            else:
                print(f"{sectorMask[i,j]:01d}", end="")
        print("")

def countEachMaskList(polarMask, N):
    maskCountList=[-1,]*N
    for i in range(N):
        maskCountList[i] = np.count_nonzero(polarMask==i)
    return maskCountList

def getSectorMaskList(polarMask, N):
    sectorMaskList=[]
    for i in range(N):
        sectorMaskList.append((polarMask == i).astype(np.int))
    return sectorMaskList


def main():
    thicknessList = glob.glob(inputDir + f"/*{inputFileSuffix}")
    thicknessList.sort()
    nVolumes = len(thicknessList)
    print(f"total {nVolumes} volumes")

    if not os.path.exists(outputDir):
        os.makedirs(outputDir)  # recursive dir creation

    # generate polarmap and sectorMask
    polarMap = generatePolarMap(H,W)
    ODPolarMask = get9SectorsMask(polarMap, "OD")
    ODCountMask = countEachMaskList(ODPolarMask, nSectors)
    ODSectorMaskList = getSectorMaskList(ODPolarMask, nSectors)
    OSPolarMask = get9SectorsMask(polarMap, "OS")
    OSCountMask = countEachMaskList(OSPolarMask, nSectors)
    OSSectorMaskList = getSectorMaskList(OSPolarMask, nSectors)
    print(f"OD 9-sector Mask with image size {H}x{W}:")
    printSectorMask(ODPolarMask)
    print(f"OS 9-sector Mask with image size {H}x{W}:")
    printSectorMask(OSPolarMask)

    # generate 9x9 sector average values
    for thickessPath in thicknessList:
        outputFilename = os.path.basename(thickessPath)
        if "_OD_" in outputFilename:
            polarMask = ODPolarMask
            countMask = ODCountMask
            sectorMaskList = ODSectorMaskList
        elif "_OS_" in outputFilename:
            polarMask = OSPolarMask
            countMask = OSCountMask
            sectorMaskList = OSSectorMaskList
        else:
            print("Error: file name can not recognize OS/OD")
            assert False

        outputFilename = outputFilename.replace(inputFileSuffix, outputFileSuffix)
        outputPath = os.path.join(outputDir, outputFilename)

        sectorMap = np.empty((nLayers, nSectors), dtype=np.float)
        # read volume
        thicknessMap = np.load(thickessPath)  # BxHxW
        assert (nLayers, H, W) == thicknessMap.shape
        for i in range(nSectors):
            sectorMask = sectorMaskList[i]
            for layer in range(nLayers):
                maskedValues = thicknessMap[layer,:,:] * sectorMask
                sectorMap[layer, i] = maskedValues.sum()/countMask[i]

        np.save(outputPath, sectorMap)


    print(f"=============End of generate 9x9 sector map ===============")

if __name__ == "__main__":
    main()