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
npyFile = '/home/hxie1/data/OvarianCancerCT/pixelSize223/numpy/06167597.npy'
image3d = np.load(npyFile)
centerSliceIndex = image3d.shape[0]//2
midSlice = image3d[centerSliceIndex]
print(f'mid slice after load: mean = {np.mean(midSlice):.8f}, std = {np.std(midSlice):.8f}, min= {np.min(midSlice):.8f}, max = {np.max(midSlice):.8f}')
display2DImage(midSlice, 'midSlice')

print('=====end of draw====')

"""



