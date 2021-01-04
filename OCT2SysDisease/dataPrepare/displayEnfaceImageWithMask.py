# visualize en-face image with Mask

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

outputDir = "/home/hxie1/data/temp"

def main():
    if len(sys.argv) != 3:
        print("Error: input parameters error.")
        print(f"{sys.argv[0]}  fullPathOfEnfaceVolume maskFullPath")
        return -1
    enfaceVolumePath = sys.argv[1]
    maskPath = sys.argv[2]

    enfaceVolume = np.load(enfaceVolumePath)
    N,H,W = enfaceVolume.shape
    print(f"enface volume: N={N}, H={H}, W={W}")

    mask = np.load(maskPath)
    assert (N,H,W) == mask.shape

    basename = os.path.basename(enfaceVolumePath)
    volumename, ext = os.path.splitext(basename)

    assert N == 9
    #surfaceNames = ['ILM', 'RNFL-GCL', 'GCL-IPL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE', 'OB_RPE']  # 10 surfaces.
    layerNames = ['RNFL', 'GCL', 'IPL', 'INL', 'OPL', 'ONL','ELM','PR', 'RPE']

    '''
    image dispaly in 3x3 order like below:
    'RNFL', 'GCL', 'IPL',
    'INL', 'OPL', 'ONL',
    'ELM','PR', 'RPE'
    '''

    if "thickness" in volumename:
        maskname = "thickness_mask"
    else:
        maskname = "texture_mask"

    volume_name_list= [
        [enfaceVolume, volumename],
        [mask, maskname],
        [enfaceVolume * mask, volumename+"_mask"],
    ]

    for (volume, outputFilename) in volume_name_list:
        # display original image
        f = plt.figure(frameon=False)
        DPI = f.dpi
        if W >=9*H:
            rowSubplot = N  # for wide image
            colSubplot = 1
        else:
            rowSubplot = 3  # for 31x25 enface images
            colSubplot = 3
        f.set_size_inches(W * colSubplot / float(DPI), H * rowSubplot / float(DPI))

        plt.margins(0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

        textLocx = int(W*0.01)
        textLocy = int(H*0.3)

        for i in range(N):
            subploti = plt.subplot(rowSubplot, colSubplot, i+1)
            subploti.imshow(volume[i,], cmap='gray')
            subploti.axis('off')
            if W >= 9*H:
                subploti.text(textLocx, textLocy, layerNames[i], fontsize=8)

        outputFilePath = os.path.join(outputDir, outputFilename+ f"_{N}x{H}x{W}.png")
        plt.savefig(outputFilePath, dpi='figure', bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"image file output at {outputFilePath}")

    #

if __name__ == "__main__":
    main()
