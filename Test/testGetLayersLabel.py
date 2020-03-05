import torch
import sys
sys.path.append("../OCTMultiSurfaces/network")
from OCTAugmentation import *


def main():
    surfaces = torch.tensor([[3.5, 4, 7, 10],[2, 3, 6, 8],[4, 5, 9, 10]]).t()
    Height = 15
    print(f"surfaces=\n{surfaces}")
    layers = getLayerLabels(surfaces,Height)
    print(f"layer=\n{layers}")



if __name__ == "__main__":
    main()