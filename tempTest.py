
from DataMgr import DataMgr
import numpy as np

prediction= np.array([1,0,1,0,1,0])
label = np.array([1,1,1,0,0,0])

TNR = DataMgr.getTNR(prediction, label)

print(TNR)
