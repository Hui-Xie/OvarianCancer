# measure max Rift

import numpy as np

# all label files
labelFiles=["/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/training/surfaces_CV0.npy",\
            "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/validation/surfaces_CV0.npy",\
            "/home/hxie1/data/OCT_Tongren/numpy/10FoldCVForMultiSurfaceNet/test/surfaces_CV0.npy"]

for i, labelFile in enumerate(labelFiles):
    surfaces = np.load(labelFile)  # BxNxW
    R = surfaces[:,1:,:] - surfaces[:,0:-1,:]  # size: Bx(N-1)xW
    Rs = np.concatenate((Rs,R), axis=0) if 0 !=i else R

print(f"Rs.shape= {Rs.shape}")
maxRift = np.amax(Rs)

print(f"maxRift= {maxRift}")

