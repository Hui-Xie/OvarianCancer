# read Numpy file

filePath = "/home/hxie1/data/BES_3K/numpy/W512/test/images_97.npy"

import numpy as np

images = np.load(filePath)
print(f"images.shape = {images.shape}")

# some voluem merge
# images.shape = (1023, 496, 512)
