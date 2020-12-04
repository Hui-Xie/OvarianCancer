# statistics thickness map

import os
import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Error: input parameters error.")
        print(f"{sys.argv[0]}  fullPathOfEnfaceVolume")
        return -1
    thicknessMapPath = sys.argv[1]

    thicknessMap = np.load(thicknessMapPath)
    C,H,W = thicknessMap.shape
    print(f"thicknessMap: C={C}, H={H}, W={W}")

    mean, std = np.mean(thicknessMap), np.std(thicknessMap)

    print(f"{thicknessMapPath}:\n mean= {mean}, std={std}")

if __name__ == "__main__":
    main()
