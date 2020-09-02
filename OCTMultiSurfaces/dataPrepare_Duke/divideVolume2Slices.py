# divide volumes into slices
volumesDir = "/home/hxie1/data/OCT_Duke/numpy/validation"
srcTag = "/numpy/"
dstTag = "/numpy_slices/" # outputSlicesDir = "/home/hxie1/data/OCT_Duke/numpy_slices/****"

patientListPath = volumesDir + "/patientList.txt"

S = 51  # the number of slices per volume
epsilon = 1e-8


import numpy as np
import os

# read all volumes list
with open(patientListPath, 'r') as f:
    volumesID = f.readlines()
volumesID = [item[0:-1] for item in volumesID]
volumesList = volumesID.copy()
labelsList = [item.replace("_images.npy", "_surfaces.npy") for item in volumesID]

# read volumes
nVolumes = len(volumesList)
for n in range(nVolumes):
    # image uses float32
    images = np.load(volumesList[n]) # S, H, W
    # image has done with normalization  in convertData.py
    '''
    std = np.std(images, axis=(1, 2))
    mean = np.mean(images, axis=(1, 2))
    for s in range(S):
        images[s] = (images[s] - mean[s])/(std[s]+epsilon)
    '''
    labels = np.load(labelsList[n])# S, num_surface, W

    # divide into slices
    imagesRoot, imageExt = os.path.splitext(volumesList[n])
    imagesRoot = imagesRoot.replace(srcTag,dstTag)
    labelRoot, labelExt = os.path.splitext(labelsList[n])
    labelRoot = labelRoot.replace(srcTag, dstTag)
    for s in range(S):
        np.save(imagesRoot + f"_s{s:02d}"+imageExt, images[s])
        np.save(labelRoot + f"_s{s:02d}"+ labelExt, labels[s])


outputSlicesDir = volumesDir.replace(srcTag, dstTag)
print(f"Now all slides in {outputSlicesDir}")
