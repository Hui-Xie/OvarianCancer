# visualize single en-face image with boudanray of 9 sectors.

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

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
    DPI = f.dpi
    f.set_size_inches(W / float(DPI), H / float(DPI))

    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

    plt.imshow(enfaceImage, cmap='gray')
    plt.axis('off')

    if displaySectors:
        D = min(H,W)
        # plt.Circle((H//2, W//2), )

    outputFilePath = os.path.join(outputDir, imagename+ f"_{H}x{W}.png")
    plt.savefig(outputFilePath, dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"image file output at {outputFilePath}")

if __name__ == "__main__":
    main()
