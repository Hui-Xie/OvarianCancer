
predictNumpyDir = "/home/hxie1/data/OCT_Beijing/netParameters/SurfNSBNet20191129/CV5/predict"
predictImageDir = "/home/hxie1/data/OCT_Beijing/netParameters/SurfNSBNet20191129/CV5/predict/visualImages"

numPatients = 8
numSlices = 31

import numpy as np
import os
import matplotlib.pyplot as plt
import glob


def main():
    dataList = glob.glob(predictNumpyDir + f"/*.npy")
    dataList.sort()

    if not os.path.isdir(predictImageDir):
        os.mkdir(predictImageDir)

    N = len(dataList)
    for i in range(0, N, 5):
        '''
        example: 
            4959_OD_28688_OCT31_Image.npy
            4959_OD_28688_OCT31_sf00_GT.npy
            4959_OD_28688_OCT31_sf01_GT.npy
            4959_OD_28688_OCT31_sf01_Pred.npy
            4959_OD_28688_OCT31_sf02_GT.npy

        '''
        imageFile = dataList[i]
        gtn1File = dataList[i+1]
        gtFile = dataList[i+2]
        predFile = dataList[i+3]
        gtp1File = dataList[i+4]

        s = os.path.basename(imageFile)
        patientIDSlice = s[0:s.rfind('_')]

        image = np.load(imageFile)
        gt = np.load(gtFile)
        gt_n1 = np.load(gtn1File)
        gt_p1 = np.load(gtp1File)
        pred = np.load(predFile)
        (H,W) = image.shape

        f = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.plot(range(0, W), gt, 'g,')
        plt.plot(range(0, W), gt_n1, 'b,')
        plt.plot(range(0, W), gt_p1, 'y,')
        plt.plot(range(0, W), pred, 'r,')

        titleName = f"{patientIDSlice}, B:GT-1, G:GT, Y:GT+1, R:Pred"
        plt.title(titleName)

        plt.savefig(os.path.join(predictImageDir, f"{patientIDSlice}.png"))
        plt.close()

if __name__ == "__main__":
    main()
