# visualize single en-face image with boudanray of 9 sectors.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
outputDir = "/home/hxie1/data/temp"

displaySectors = True

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        print(f"{sys.argv[0]}  fullPathOfEnfaceImage")
        return -1
    enfaceImagePath = sys.argv[1]

    enfaceImage = np.load(enfaceImagePath)
    N, H,W = enfaceImage.shape
    assert N==1
    enfaceImage = enfaceImage.squeeze(axis=0)
    print(f"enface Image: H={H}, W={W}")

    basename = os.path.basename(enfaceImagePath)
    imagename, ext = os.path.splitext(basename)

    f = plt.figure(frameon=False)
    ax = plt.gca()
    DPI = f.dpi
    f.set_size_inches(W / float(DPI), H / float(DPI))

    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

    plt.imshow(enfaceImage, cmap='gray')
    plt.axis('off')



    outputName = imagename+ f"_{H}x{W}.png"
    if displaySectors:
        outputName = imagename+ f"_9sector_{H}x{W}.png"
        a = math.sqrt(2)/2.0
        D = min(H,W)
        r = D//2
        circle1= plt.Circle((W//2, H//2), D//6, color='r', fill=False, linewidth=0.1)
        circle2 = plt.Circle((W // 2, H // 2), D // 3, color='r', fill=False, linewidth=0.1)
        circle3 = plt.Circle((W // 2, H // 2), D // 2, color='r', fill=False, linewidth=0.1)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        x1,y1 = [W//2+r/3*a, W//2+r*a],[H//2+ r/3*a, H//2+r*a]  # 45 degree
        x2,y2 = [W//2-r/3*a, W//2-r*a],[H//2+ r/3*a, H//2+r*a]  # 134 degree
        x3,y3 = [W//2-r/3*a, W//2-r*a],[H//2- r/3*a, H//2-r*a]  # -135 degree
        x4,y4 = [W//2+r/3*a, W//2+r*a],[H//2- r/3*a, H//2-r*a]  # -145 degree
        plt.plot(x1,y1,x2,y2, x3,y3, x4,y4, color='r', linewidth=0.1)



    outputFilePath = os.path.join(outputDir, outputName)
    plt.savefig(outputFilePath, dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"image file output at {outputFilePath}")

if __name__ == "__main__":
    main()
