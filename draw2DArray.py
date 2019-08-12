# display 2D image

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def display2DImage(array2d, title):
    plt.imshow(array2d)
    plt.colorbar()
    plt.title(title)
    plt.show()

"""

# test:
nrrdFile = '/home/hxie1/data/OvarianCancerCT/pixelSize223/nrrd/06167597_CT.nrrd'
npyFile = '/home/hxie1/temp/06167597.npy'
Z,Y,X = 231,251,251
image = sitk.ReadImage(nrrdFile)
image3d = sitk.GetArrayFromImage(image)

# window level image into [0,300]
image3d = np.clip(image3d, 0, 300)
image3d = image3d.astype(float)

midSliceBeforeNorm = image3d[image3d.shape[0]//2,]
print(f'midSliceBeforeNorm: mean = {np.mean(midSliceBeforeNorm):.8f}, std = {np.std(midSliceBeforeNorm):.8f}, min= {np.min(midSliceBeforeNorm):.8f}, max= {np.max(midSliceBeforeNorm):.8f}')

# normalize image with std  for each slice
shape = image3d.shape
for i in range(shape[0]):
    slice = image3d[i,]
    mean = np.mean(slice)
    std  = np.std(slice)
    if 0 != std:
        slice = (slice -mean)/std
    else:
        slice = slice -mean
    image3d[i,] = slice

midSliceBeforeAssem = image3d[image3d.shape[0]//2,]
print(f'midSliceBeforeAssem: mean = {np.mean(midSliceBeforeAssem):.8f}, std = {np.std(midSliceBeforeAssem):.8f}, min= {np.min(midSliceBeforeAssem):.8f}, max= {np.max(midSliceBeforeAssem):.8f}')

# assemble in fixed size[231,251,251] in Z,Y, X direction
z,y,x = image3d.shape
wall = np.zeros((Z,Y,X), dtype=np.float)
if z<Z:
    Z1 = (Z-z)//2
    Z2 = Z1+z
    z1 = 0
    z2 = z1+z
else:
    Z1 = 0
    Z2 = Z1+Z
    z1 = (z-Z)//2
    z2 = z1+Z

if y < Y:
    Y1 = (Y - y) // 2
    Y2 = Y1 + y
    y1 = 0
    y2 = y1 + y
else:
    Y1 = 0
    Y2 = Y1 + Y
    y1 = (y - Y) // 2
    y2 = y1 + Y

if x < X:
    X1 = (X - x) // 2
    X2 = X1 + x
    x1 = 0
    x2 = x1 + x
else:
    X1 = 0
    X2 = X1 + X
    x1 = (x - X) // 2
    x2 = x1 + X

wall[Z1:Z2, Y1:Y2,X1:X2] = image3d[z1:z2, y1:y2, x1:x2]

midSliceWall = wall[wall.shape[0]//2,]
print(f'mid Slice Wall: mean = {np.mean(midSliceWall):.8f}, std = {np.std(midSliceWall):.8f}, min= {np.min(midSliceWall):.8f}, max = {np.max(midSliceWall):.8f}')


np.save(npyFile, wall)
#np.save(os.path.join(outputLabelsDir, patientID + '.npy'), label3d)



#file = '/home/hxie1/data/OvarianCancerCT/pixelSize223/numpy/06167597.npy'
image3d = np.load(npyFile)
centerSliceIndex = image3d.shape[0]//2
midSlice = image3d[centerSliceIndex]
print(f'mid slice after load: mean = {np.mean(midSlice):.8f}, std = {np.std(midSlice):.8f}, min= {np.min(midSlice):.8f}, max = {np.max(midSlice):.8f}')
display2DImage(midSlice, 'midSlice')

print('=====end of draw====')

"""

