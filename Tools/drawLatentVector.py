latentDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223/latent"
responsePath =  "/home/hxie1/data/OvarianCancerCT/patientResponseDict.json"

lenLV = 512

import numpy as np
from colorama import Fore

import sys
sys.path.append("..")
from FilesUtilities import *
fileList = getFilesList(latentDir, ".npy")
N = len(fileList)

import json
with open(responsePath) as f:
    responseDict = json.load(f)


import matplotlib.pyplot as plt

# draw Line figures.
f1 = plt.figure(1)
xAxis = np.array(range(0,lenLV), dtype=int)
blue = 50   # for response 1
red = 50    # for response 0
subplot0 = plt.subplot(2,1,1)
subplot1 =  plt.subplot(2,1,2)
for i in range(N):
    patientID = getStemName(fileList[i], ".npy")[0:8]
    latentV = np.load(fileList[i])

    response = responseDict[patientID]
    if response == 1:
        blue += 1
        color = '#'+hex(blue)[2:].zfill(6)
        subplot1.plot(xAxis, latentV, color=color)
        print (Fore.BLUE + str(latentV.tolist()))
    else:
        red += 1
        color = '#'+(hex(red)[2:]+"0000")
        subplot0.plot(xAxis, latentV, color=color)
        print(Fore.RED + str(latentV.tolist()))

subplot0.set_xlabel('Element Position')
subplot0.set_ylabel('Element Value')
subplot0.set_title('Response 0')

subplot1.set_xlabel('Element Position')
subplot1.set_ylabel('Element Value')
subplot1.set_title('Response 1')

plt.tight_layout()

print(f"red has {red-50} lines, and blue has {blue-50} lines.")

plt.savefig(os.path.join(latentDir, f"latentVLine.png"))


# draw circles
f2 = plt.figure(2)
theta = np.array(range(0,lenLV), dtype=int)* 2*np.pi/lenLV
blue = 50   # for response 1
red = 50    # for response 0
subplot0 = plt.subplot(2,1,1, projection="polar")
subplot1 =  plt.subplot(2,1,2, projection="polar")
for i in range(N):
    patientID = getStemName(fileList[i], ".npy")[0:8]
    latentV = np.load(fileList[i])
    latentV = np.abs(latentV)

    response = responseDict[patientID]
    if response == 1:
        blue += 1
        color = '#'+hex(blue)[2:].zfill(6)
        subplot1.plot(theta, latentV, color=color)

    else:
        red += 1
        color = '#'+(hex(red)[2:]+"0000")
        subplot0.plot(theta, latentV, color=color)


subplot0.set_title('Response 0', pad=20)
subplot1.set_title('Response 1', pad=20)

plt.tight_layout()

plt.savefig(os.path.join(latentDir, f"latentVCircle.png"))

plt.show()