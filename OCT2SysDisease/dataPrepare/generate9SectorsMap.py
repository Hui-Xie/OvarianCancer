# generate 9 sector thickness map for the 5th thickness map

inputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/5thThickness_1x31x25"
outputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/5thThickness_9Sectors"

H = 31  # odd number
W = 25  # odd number
c  =5   # the 5th thickness map
inputFileSuffix = "_Volume_thickness_enface.npy"
outputFileSuffix = "_Volume_5thickness_9sectors.npy"
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
    recMap[1,] = np.repeat(np.arange(-(H // 2), H // 2 + 1).reshape(H, 1), W, axis=1)
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
    sectorMask = np.ones((H,W), dtype=np.int)*(-1)
    for i in range(H):
        for j in range(W):
            p = polarMap[0,i,j]
            theta = polarMap[1,i,j]
            if p <= stepR:
                sectorMask[i,j] = 0
            elif -Pi/4 <= theta < Pi/4:  # 1,3,5,7
                if stepR < p <= 2*stepR:
                    if OS_OD =="OD":
                        sectorMask[i, j] = 1
                    else:
                        sectorMask[i, j] = 3
                if 2*stepR < p <= 3*stepR:
                    if OS_OD =="OD":
                        sectorMask[i, j] = 5
                    else:
                        sectorMask[i, j] = 7

            elif Pi/4 <= theta < Pi*3/4: # 2,6
                if stepR < p <= 2 * stepR:
                    sectorMask[i, j] = 2
                if 2 * stepR < p <= 3 * stepR:
                    sectorMask[i, j] = 6

            elif (Pi*3/4 <= theta)  or (theta < -Pi*3/4): # 1,3,5,7
                if stepR < p <= 2 * stepR:
                    if OS_OD == "OD":
                        sectorMask[i, j] = 3
                    else:
                        sectorMask[i, j] = 1
                if 2 * stepR < p <= 3 * stepR:
                    if OS_OD == "OD":
                        sectorMask[i, j] = 7
                    else:
                        sectorMask[i, j] = 5
            elif -Pi*3/4 <= theta < -Pi/4: #4, 8
                if stepR < p <= 2 * stepR:
                    sectorMask[i, j] = 4
                if 2 * stepR < p <= 3 * stepR:
                    sectorMask[i, j] = 8
            else:
                print(f"Error in phase from polarmap")
                assert False
    return sectorMask

def printSectorMask(sectorMask):
    print("========sectorMask=========")
    H,W = sectorMask
    for i in range(H):
        for j in range(W):
            if sectorMask[i,j]==-1:
                print(" ", end="")
            else:
                print(f"{sectorMask[i,j]:01d}", end="")
        print("")


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
    OSPolarMask = get9SectorsMask(polarMap, "OS")
    print(f"OD 9-sector Mask with image size {H}x{W}:")
    printSectorMask(ODPolarMask)
    print(f"OS 9-sector Mask with image size {H}x{W}:")
    printSectorMask(OSPolarMask)

    '''
    
    
    for thickessPath in thicknessList:
        outputFilename = os.path.basename(thickessPath)
        outputFilename = outputFilename.replace(inputFileSuffix, outputFileSuffix)
        outputPath = os.path.join(outputDir, outputFilename)

        newVolume = np.empty((2, 31, 25), dtype=np.float)
        # read volume
        thicknessVolume = np.load(thickessPath)  # BxHxW
        assert (9, 31, 25) == thicknessVolume.shape

        textureVolume = np.load(texturePath)
        assert (9, 31, 25) == textureVolume.shape

        newVolume[0,] = thicknessVolume[c1,]
        newVolume[1,] = textureVolume[c2,]
        np.save(outputPath, newVolume)
    '''

    print(f"=============End of generate 9-sector map ===============")

if __name__ == "__main__":
    main()