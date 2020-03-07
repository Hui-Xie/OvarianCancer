import torch
import sys
sys.path.append("../OCTMultiSurfaces/network")
from OCTAugmentation import getLayerLabels, layerProb2SurfaceMu

sys.path.append("../framework")
from CustomizedLoss import logits2Prob


def main():
    surfaces = torch.tensor([[3.5, 4, 7, 10],[2, 2, 6, 8],[4, 5, 9, 10]]).t()
    Height = 15
    N,W = surfaces.shape
    print(f"surfaces=\n{surfaces}")
    layers = getLayerLabels(surfaces,Height)
    print(f"layer=\n{layers}")

    layers = layers.view(1,15,3) # B,H,W
    B,H,W = layers.shape

    layerProb = torch.zeros((B,N+1,H,W), dtype=torch.float)
    randB1HW = torch.rand((B,1,H,W))
    layerProb.scatter_(1, layers.unsqueeze(dim=1),  randB1HW)

    layerProb = logits2Prob(layerProb, dim=1)

    surfaceMu, surfaceConf = layerProb2SurfaceMu(layerProb)

    print(f"surfaceMu = \n{surfaceMu}")
    print(f"surfaceConf= \n{surfaceConf}")





if __name__ == "__main__":
    main()