# use height-direction thin-plate-spline (TPs) to smooth surface.

from utilities import  getSurfacesArray
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import os
import random



extractIndexs = (0, 1, 3, 5, 6, 10) # extracted surface indexes from original 11 surfaces.

H = 1024
N = len(extractIndexs)
W = 200  # target image width
B = 200 # number of slices
surfaceIndex = 1
C = 1000 # the number of random chosed control points for TPS. C is a multiple of 8.


# surfacesXmlPath ="/home/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set1/PVIP2-4060_Macular_200x200_8-25-2009_11-55-11_OD_sn16334_cube_z/PVIP2-4060_Macular_200x200_8-25-2009_11-55-11_OD_sn16334_cube_z_Surfaces_Iowa.xml"
# outputImageDir = "/home/hxie1/temp"
surfacesXmlPath = "/home/sheen/temp/PVIP2-4074_Macular_200x200_11-7-2013_8-14-8_OD_sn26558_cube_z/PVIP2-4074_Macular_200x200_11-7-2013_8-14-8_OD_sn26558_cube_z_Surfaces_Iowa.xml"
outputImageDir = "/home/sheen/temp"

basename = os.path.basename(surfacesXmlPath)
basename = basename[0:basename.rfind("_Surfaces_Iowa.xml")]

# read xml file
surfaces = getSurfacesArray(surfacesXmlPath)  # size: SxNxW, where N is number of surfacres.
surfaces = surfaces[:, extractIndexs, :]   #  extract 6 surfaces (0, 1, 3, 5, 6, 10)
# its surface names: ["ILM", "RNFL-GCL", "IPL-INL", "OPL-HFL", "BMEIS", "OB_RPE"]
B1, curN, W1 = surfaces.shape  # BxNxW
assert N == curN
assert B == B1
assert W == W1

surface = surfaces[:,surfaceIndex,:]  # choose surface 1, size: BxW
coordinateSurface = np.mgrid[0:B, 0:W]
coordinateSurface = coordinateSurface.reshape(2, -1).T  # size (BxW) x2  in 2 dimension.

# random sample C control points in the original surface of size BxW, with a repeatable random.
randSeed = 20217  # fix this seed for ground truth and prediction.
random.seed(randSeed)
P = list(range(0, B*W))
chosenList = [0,]*C
# use random.sample to choose unique element without replacement.
chosenList[0:C//8] = random.sample(P[0:W*B//4],k= C//8)
chosenList[C//8:C//2] = random.sample(P[W*B//4: W*B//2],k= 3*C//8)
chosenList[C//2:7*C//8] = random.sample(P[W*B//2: W*3*B//4],k= 3*C//8)
chosenList[7*C//8: C] = random.sample(P[W*3*B//4: W*B],k= C//8)
chosenList.sort()

controlCoordinates = coordinateSurface[chosenList,:]
controlValues = surface.flatten()[chosenList,]


interpolator = RBFInterpolator(controlCoordinates, controlValues, neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
newSurface = interpolator(coordinateSurface).reshape(B,W)

max = np.absolute(newSurface-surface).max()
print(f"max modification: = {max}")
avg = np.average(newSurface-surface)
print(f"average modification: = {avg}")
'''
max modification: = 9.183762819906661
average modification: = -0.007614259520222018
'''


# display before TPS and after TPS image.

f = plt.figure(frameon=False)
DPI = 100
rowSubplot = 1
colSubplot = 2
f.set_size_inches(W * colSubplot / float(DPI), H * rowSubplot / float(DPI))

plt.margins(0)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # very important for erasing unnecessary margins.

subplot1 = plt.subplot(rowSubplot, colSubplot, 1)
subplot1.imshow(surface, cmap='viridis')
subplot1.axis('off')

subplot2 = plt.subplot(rowSubplot, colSubplot, 2)
subplot2.imshow(newSurface, cmap='viridis')
subplot2.axis('off')

# plt.colorbar(subplot2, ax=subplot2.axis)

curImagePath = os.path.join(outputImageDir, basename+f"_s{surfaceIndex:03d}_surfaceHotmap.png")

plt.savefig(curImagePath, dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()
