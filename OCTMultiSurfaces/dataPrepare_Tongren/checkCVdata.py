# check generated cross validation image and surface data with specific slice.

imagesPath = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/validation/images_CV6.npy"
surfacesPath = "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/validation/surfaces_CV6.npy"
displayIndex = 122

import numpy as np
import matplotlib.pyplot as plt

images = np.load(imagesPath)
surfaces = np.load(surfacesPath)
S,H,W = images.shape
S1,NumSurface,W1 = surfaces.shape
assert S==S1 and W==W1
print (f"S={S}, H={H}, W={W}, NumSurfaces={NumSurface}")

f = plt.figure()
plt.imshow(images[displayIndex,], cmap='gray')
for s in range(0, NumSurface):
    plt.plot(range(0,W), surfaces[displayIndex,s,:], linewidth=0.7)

plt.show()

plt.close()



