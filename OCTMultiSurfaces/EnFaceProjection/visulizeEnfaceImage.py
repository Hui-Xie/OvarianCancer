# visulize en-face image

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

outputDir = "/home/hxie1/data/temp"

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        print(f"{sys.argv[0]}  fullPathOfEnfaceVolume")
        return -1
    enfaceVolumePath = sys.argv[1]

    enfaceVolume = np.load(enfaceVolumePath)
    N,B,W = enfaceVolume.shape
    print(f"enface volume: N={N}, B={B}, W={W}")

    basename = os.path.basename(enfaceVolumePath)
    volumename, ext = os.path.splitext(basename)

    assert N == 9
    #surfaceNames = ['ILM', 'RNFL-GCL', 'GCL-IPL', 'IPL-INL', 'INL-OPL', 'OPL-HFL', 'BMEIS', 'IS/OSJ', 'IB_RPE', 'OB_RPE']  # 10 surfaces.
    layerNames = ['RNFL', 'GCL', 'IPL', 'INL', 'OPL', 'ONL','ELM','PR', 'RPE']


    f = plt.figure(frameon=False)
    DPI = f.dpi
    rowSubplot = 3
    colSubplot = 3
    f.set_size_inches(W * colSubplot / float(DPI), B * rowSubplot / float(DPI))

    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

    textLocx = int(W*0.5)
    textLocy = int(B*1.0)

    for i in range(N):
        subploti = plt.subplot(rowSubplot, colSubplot, i+1)
        subploti.imshow(enfaceVolume[i,], cmap='gray')
        subploti.axis('off')
        subploti.text(textLocx, textLocy, layerNames[i], fontsize=8)

    outputFilePath = os.path.join(outputDir, volumename+ ".png")
    plt.savefig(outputFilePath, dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"image file output at {outputFilePath}")

if __name__ == "__main__":
    main()
