# generate 9 sector thickness map for the 5th thickness map

inputDir = "/home/hxie1/data/BES_3K/W512NumpyVolumes/log/SurfacesNet/expBES3K_20201126A_genXml/testResult/5thThickness_1x31x25"

H = 31  # odd number
W = 25  # odd number


import numpy as np
import cmath
import math

Pi = math.pi

def generatePolarMask(H, W):
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
    for i in range(W):
        for j in range(H):
            polarMap[0,i,j], polarMap[1,i,j] = cmath.polar(complex(recMap[0,i,j], recMap[1,i,j])) # phase is in range(-Pi,Pi]
    return polarMap



def main():
    pass

if __name__ == "__main__":
    main()