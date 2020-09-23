import torch

x = torch.rand(1,2,3,3)
print(f"x= \n{x}")
batchNorm = torch.nn.BatchNorm2d(2, eps=0)
yBatchNorm = batchNorm(x)
# conclusion: batchNorm2d does norm an all (each feature * all batch) dimension.
print(f"yBatchNorm = \n{yBatchNorm}")

layerNorm = torch.nn.LayerNorm([2,3,3], elementwise_affine=False)
yLayerNorm = layerNorm(x)
print(f"yLayerNorm = \n{yLayerNorm}")
# connclusion: layerNorm does norm on all elements on specified dimensions.


print("============")
