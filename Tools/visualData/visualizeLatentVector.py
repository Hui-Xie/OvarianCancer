latentDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223/latent/latent_20190902_175550"
responsePath =  "/home/hxie1/data/OvarianCancerCT/patientResponseDict.json"

lenLV = 512

import numpy as np
from colorama import Fore, Style

import sys
sys.path.append("..")
from utilities.FilesUtilities import *
fileList = getFilesList(latentDir, ".npy")
N = len(fileList)

import json
with open(responsePath) as f:
    responseDict = json.load(f)

# statistics 0 and 1
nCount0 = 0
nCount1 = 0
for i in range(N):
    patientID = getStemName(fileList[i], ".npy")[0:8]
    response = responseDict[patientID]
    if 0 == response:
        nCount0 +=1
    else:
        nCount1 +=1
print(f"\nFor all latent vectors, response 1 has {nCount1}, and response 0 has {nCount0}.\n")

# read all latent vector into arrays.
array0 = np.empty([lenLV, nCount0], dtype=np.float)
array1 = np.empty([lenLV, nCount1], dtype=np.float)
patientID0List = []
patientID1List = []
c0=0
c1=0
for i in range(N):
    patientID = getStemName(fileList[i], ".npy")[0:8]
    latentV = np.load(fileList[i])
    response = responseDict[patientID]
    if 0 == response:
        array0[:,c0] = latentV
        patientID0List.append(patientID)
        c0 +=1
    else:
        array1[:,c1] = latentV
        patientID1List.append(patientID)
        c1 +=1

# print out latent vectors

#for i in range(0, nCount0): # print part
#    print (f"\ni={i}  patientID: {patientID0List[i]}")
#    print(Fore.RED + str(array0[:, i].tolist()))
#for i in range(0, nCount1):  # print part
#    print(f"\ni={i}   patientID: {patientID1List[i]}")
#    print(Fore.BLUE + str(array1[:, i].tolist()))

# print(Style.RESET_ALL)
#print(f"For all latent vectors, response 1 has {nCount1}, and response 0 has {nCount0}.\n")

# draw latent vectors
import matplotlib.pyplot as plt

'''
For different color for different curve.
        red = 50    # for response 0
        blue = 50   # for response 1
        
        blue += 1
        color = '#'+hex(blue)[2:].zfill(6)
        subplot1.plot(xAxis, latentV, color=color)
        
        red += 1
        color = '#'+(hex(red)[2:]+"0000")
        subplot0.plot(xAxis, latentV, color=color)
        
'''
# draw Line figures.
f1 = plt.figure(1)
xAxis = np.array(range(0,lenLV), dtype=int)


subplot0 = plt.subplot(2,1,1)
for i in range(nCount0):
    subplot0.scatter(xAxis, array0[:,i],s=1)
subplot0.set_xlabel('Element Position')
subplot0.set_ylabel('Element Value')
subplot0.set_title('Response 0')

subplot1 =  plt.subplot(2,1,2)
for i in range(nCount1):
    subplot1.scatter(xAxis, array1[:,i],s=1)
subplot1.set_xlabel('Element Position')
subplot1.set_ylabel('Element Value')
subplot1.set_title('Response 1')

plt.tight_layout()

plt.savefig(os.path.join(latentDir, f"latentVLine.png"))

# draw circles
f2 = plt.figure(2)
theta = np.array(range(0,lenLV), dtype=int)* 2*np.pi/lenLV

subplot0 = plt.subplot(2,1,1, projection="polar")
for i in range(nCount0):
    subplot0.scatter(theta, array0[:,i],s=1)
subplot0.set_title('Response 0', pad=20)

subplot1 =  plt.subplot(2,1,2, projection="polar")
for i in range(nCount1):
    subplot1.scatter(theta, array1[:,i],s=1)
subplot1.set_title('Response 1', pad=20)

plt.tight_layout()

plt.savefig(os.path.join(latentDir, f"latentVCircle.png"))

# draw mean, std curve
f3 = plt.figure(3)
array0mean = np.mean(array0, axis=1)
array0std  = np.std(array0, axis=1)

array1mean = np.mean(array1, axis=1)
array1std  = np.std(array1, axis=1)

subplot0 = plt.subplot(2,1,1)
subplot0.scatter(xAxis, array0mean, s=1)
subplot0.scatter(xAxis, array1mean, s=1 )
subplot0.set_xlabel('Element Position')
subplot0.set_ylabel('Element Value')
subplot0.legend(('mu0', 'mu1'))
subplot0.set_title('Mean between Response 0 and 1')

subplot1 = plt.subplot(2,1,2)
subplot1.scatter(xAxis, array0std,s=1)
subplot1.scatter(xAxis, array1std,s=1)
subplot1.set_xlabel('Element Position')
subplot1.set_ylabel('Element Value')
subplot1.legend(('std0', 'std1'))
subplot1.set_title('Std Deviation bewtwen Response 0 and 1')

plt.tight_layout()

plt.savefig(os.path.join(latentDir, f"latentV_mu_std.png"))

print(f"Output figures dir: {latentDir}")

# final show
plt.show()
