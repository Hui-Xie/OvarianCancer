# display the hot map of searching Lambda

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        print(sys.argv[0], "errorArrayFullPath.npy")
        return -1
    errorArrayPath = sys.argv[1] # e.g. `~/temp/muErr_predictR__lmd0_0_4.0_1.0__lmd1_0_4.0_1.0.npy
    filename = os.path.basename(errorArrayPath)
    b, ext = os.path.splitext(filename) # b = muErr_predictR__lmd0_0_4.0_1.0__lmd1_0_4.0_1.0
    lambda0Range = b[b.find("__lmd0_")+7: b.find("__lmd1_")]
    lambda1Range = b[b.find("__lmd1_")+7:]
    lambda0Range = list(lambda0Range.split('_'))
    lambda1Range = list(lambda1Range.split('_'))
    lambda0Range = [float(item) for item in lambda0Range] # min, max, step
    lambda1Range = [float(item) for item in lambda1Range] # min, max, step

    # file path for the output hot map
    a = errorArrayPath
    hotmapPath = a[:a.find(".npy")]+ "_hotmap.png"

    errorArray = np.load(errorArrayPath)
    H,W = errorArray.shape

    f = plt.figure(frameon=False)
    DPI = f.dpi
    f.set_size_inches(W / float(DPI), H/ float(DPI))

    plt.margins(0)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

    plt.imshow(errorArray, cmap='viridis')  # 'hot'
    plt.colorbar()
    plt.xlim(lambda0Range[0], lambda0Range[1]-lambda0Range[2] )
    plt.ylim(lambda1Range[0], lambda1Range[1]-lambda1Range[2])
    plt.xlabel("lambda_0")
    plt.ylabel("lambda_1")
    plt.show()

    plt.savefig(hotmapPath, dpi='figure', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"image file output at {hotmapPath}")

    print("========end=========")




if __name__ == "__main__":
    main()

