
# check numpy data and its correponding label

import sys
import numpy as np
import matplotlib.pyplot as plt

imageFile = "/home/hxie1/data/OvarianCancerCT/primaryROI/nrrd_npy/04110420.npy"
labelFile = "/home/hxie1/data/OvarianCancerCT/primaryROI/labels_npy/04110420.npy"

image = np.load(imageFile).astype(np.float32)
label = np.load(labelFile).astype(np.float32)

print(f"images size: {image.shape}")

print(f"Notes: numpy data image rotates 180 degree comparing with 3D slicer image.")

if image.shape != label.shape:
    print("Error: image.shape != label.shape")
    sys.exit(0)

s = 25
subplot1 = plt.subplot(2,2,1)
subplot1.imshow(image[s,])

subplot2 = plt.subplot(2,2,2)
subplot2.imshow(label[s,])

subplot3 = plt.subplot(2,2,3)
subplot3.imshow(label[s,])

subplot4 = plt.subplot(2,2,4)
subplot4.imshow(image[s,]+ label[s])

plt.show()