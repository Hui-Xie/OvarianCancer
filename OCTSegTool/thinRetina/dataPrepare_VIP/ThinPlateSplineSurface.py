# use height-direction thin-plate-spline (TPs) to smooth surface.

from utilities import  getSurfacesArray
import numpy as np
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import os



extractIndexs = (0, 1, 3, 5, 6, 10) # extracted surface indexes from original 11 surfaces.
outputImageDir = "/home/hxie1/temp"
H = 1024
N = len(extractIndexs)
W = 200  # target image width
B = 200 # number of slices
surfaceIndex = 1

surfacesXmlPath ="/home/hxie1/data/thinRetina/rawMhd/IOWA_VIP_25_Subjects_Thin_Retina/Graph_Search/Set1/PVIP2-4060_Macular_200x200_8-25-2009_11-55-11_OD_sn16334_cube_z/PVIP2-4060_Macular_200x200_8-25-2009_11-55-11_OD_sn16334_cube_z_Surfaces_Iowa.xml"

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
interpolator = RBFInterpolator(coordinateSurface, surface.flatten(), neighbors=None, smoothing=0.0, kernel='thin_plate_spline', epsilon=None, degree=None)
newSurface = interpolator(coordinateSurface).reshape(B,W)


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

curImagePath = os.path.join(outputImageDir, basename+f"_s{surfaceIndex:03d}_surfaceHeight.png")

plt.savefig(curImagePath, dpi='figure', bbox_inches='tight', pad_inches=0)
plt.close()
