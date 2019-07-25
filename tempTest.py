
from DataMgr import DataMgr
import numpy as np


array = np.array([[1,2,3],[2,2,2],[7,5,9]], float)

axesTuple = tuple([x for x in range(1, array.ndim)])
minx = np.min(array, axesTuple)
result = np.zeros(array.shape)
for i in range(len(minx)):
    result[i, :] = array[i, :] - minx[i]
ptp = np.ptp(array, axesTuple)  # peak to peak
with np.nditer(ptp, op_flags=['readwrite']) as it:
    for x in it:
        if 0 == x:
            x[...] = 1e-6
for i in range(len(ptp)):
    result[i, :] /= ptp[i]


print(result)


class A():
    def __init__(self):
        pass

    def __call__(self,x):
        print(f"x= {x}")

a = A();

if a:
   a(5)

a = None
if a:
    a(10)


