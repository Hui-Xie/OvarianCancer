'''
At Feb 06th, 2020:
According to notes of doctors, I plan below solution to refine the ground truth:

1   delete ill cases: #34169, and # 2579;  done
2   ROI left shifts 25 pixel for case 5363; done
3   use yaxing 20200202 updated 6 cases to replace old ground truth cases;  done;
4   delete  all surface 8 for all 47 cases;  done
5   near foveas in first 5 surface  average to erase artifacts: solve  the disorder of surface [0-5) near foveas;  done
6   sort along all columns for all patients: solve all disorder of surface 7, 8,9,  done.

After these, I will output visual images of current xml ground truth, disorder location, our adjusted result, for doctor to further review.


'''

segDir = "/home/hxie1/data/OCT_Tongren/refinedGT_20200204"
volumesDir = "/home/hxie1/data/OCT_Tongren/control"
outputDir = "/home/hxie1/data/OCT_Tongren/markError_Avg_20200206"

# glob all file

import glob
import sys

sys.path.append(".")
from FileUtilities import *
import torch
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def convertList2RangeStr(aList):
    aList.sort()
    N = len(aList)
    output = "{ "
    i = 0
    if 0 == N:
        output += " }"
    else:
        output += str(aList[i])
        i += 1
        while i < N:
            if aList[i] != aList[i - 1] + 1:
                output += f", {str(aList[i])}"
            elif i + 1 < N and aList[i] + 1 != aList[i + 1] or i + 1 == N:
                output += f"-{str(aList[i])}"
            else:  # aList[i] == aList[i-1] +1 and aList[i] +1 == aList[i+1]
                pass
            i += 1
        output += " }"
    return output


def main():
    device = torch.device('cuda:1')

    patientsList = glob.glob(segDir + f"/*_Volume_Sequence_Surfaces_Iowa.xml")
    outputFile = open(os.path.join(outputDir, "violateSeparation_20120204.txt"), "w")
    analyzeFile = open(os.path.join(outputDir, "analysis_20120204.txt"), "w")

    notes = '''
    Check whether ground truth conforms the separation constraints: h_{i+1} >= h_i, where i is surface index.
    Only check 128:640 columns with column starting index 0 for each paitent and each OCT Bscan.
    In output below , column index w mean column (w+128) in original width 768 OCT images.
         
    In output below, bscan index starts with 0 which corresponds OCT1 in the original images.\n 
    At Feb 06th, 2020:
    According to notes of doctors, I did below solution to refine the ground truth:

    1   delete ill cases: #34169, and # 2579; 
    2   ROI left shifts 25 pixel for case 5363_OD_25453;
    3   use yaxing 20200202 updated 6 cases to replace old ground truth cases;  
    4   delete  all surface 8 for all 47 cases;
    5   near foveas in first 5 surfaces,  average disorder along column to erase artifacts: solve the disorder of surface 0-5 near foveas;
    6   sort along all columns for all patients: solve all disorder of surface 7, 8,9 or other locations;

    After these, I will output visual images of current xml ground truth, disorder location, our adjusted result, for doctor to further review.

    '''

    print(notes, file=outputFile)

    print(f"This program will run about 20 mins, please wait.....")

    W = 768
    H = 496

    errorPatients = 0
    errorNum = 0
    for patientXmlPath in patientsList:
        patientSurfaceName = os.path.splitext(os.path.basename(patientXmlPath))[
            0]  # e.g. 1062_OD_9512_Volume_Sequence_Surfaces_Iowa
        patientVolumeName = patientSurfaceName[
                            0:patientSurfaceName.find("_Sequence_Surfaces_Iowa")]  # 1062_OD_9512_Volume
        patientName = patientVolumeName[0:patientVolumeName.find("_Volume")]  # 1062_OD_9512

        surfacesArray = getSurfacesArray(patientXmlPath)
        B, Num_Surfaces, W = surfacesArray.shape
        if 11 == Num_Surfaces:
            surfacesArray = np.delete(surfacesArray, 8, axis=1)  # delete surface 8
            B, Num_Surfaces, W = surfacesArray.shape
            assert B == 31 and Num_Surfaces == 10 and W == 768

        if "5363_OD_25453" == patientName:
            surfacesArray = surfacesArray[:, :, 103:615]  # left shift 25 pixels for case 5363_OD_25453
        else:
            surfacesArray = surfacesArray[:, :, 128:640]



        surfaces = torch.from_numpy(surfacesArray).to(device)
        surface0 = surfaces[:, :-1, :]
        surface1 = surfaces[:, 1:, :]
        if torch.all(surface1 >= surface0):
            continue
        else:
            errorPatients += 1
            errorLocations = torch.nonzero(surface0 > surface1)
            currentErrorNum = errorLocations.shape[0]
            errorNum += currentErrorNum

            # for exact coordinates of dislocations
            outputFile.write(
                f"\n{patientSurfaceName} violates surface separation constraints in {errorLocations.shape[0]} locations indicated by below coordinates (BScan, surface, width):\n")
            for i in range(currentErrorNum):
                if 0 != i % 10 and i != 0:
                    outputFile.write(
                        f"[{errorLocations[i, 0].item():2d},{errorLocations[i, 1].item():2d},{errorLocations[i, 2].item():3d}], ")
                else:
                    outputFile.write(
                        f"\n[{errorLocations[i, 0].item():2d},{errorLocations[i, 1].item():2d},{errorLocations[i, 2].item():3d}], ")
            outputFile.write("\n")

            # for analysis about the dislocations
            analyzeFile.write(f"\n{patientSurfaceName}:\n")
            analyzeFile.write(f"ErrorSurface, \tInvolvedBscans\n")
            surfacesSet = set(errorLocations[:, 1].tolist())
            surfaceBscanDict = {}
            for s in surfacesSet:
                surfaceBscanDict[str(s)] = set()
            for i in range(currentErrorNum):
                surfaceBscanDict[str(errorLocations[i, 1].item())].add(errorLocations[i, 0].item())
            for s in surfacesSet:
                analyzeFile.write(f"{s}, \t{convertList2RangeStr(list(surfaceBscanDict[str(s)]))}\n")

            # use average method along column to correct disorders in surface 0-4 near foveas
            surfacesCopy = torch.zeros_like(surfaces)  # surfaceCopy contain the first disorder values only, it and its next row value along column violate constraints
            surfacesCopy[:,0:-1,:] = torch.where(surface0 > surface1, surface0, torch.zeros_like(surface0))

            B,N,W = surfaces.shape
            MaxMergeSurfaces = 5
            for b in range(0, B):
                for w in range(W*2//5, W*3//5): # only focus on fovea region.
                    s = 0
                    while (s < MaxMergeSurfaces): # in fovea region, the first 5 surfaces may merge into one surface.
                        if 0 != surfacesCopy[b, s, w]:
                            n = 1  # n continuous disorder locations
                            while s + n+1 < MaxMergeSurfaces and 0 != surfacesCopy[b, s + n, w]:
                                n += 1
                            # get the average of disorder surfaces
                            correctValue = 0.0
                            for k in range(0, n+1): # n+1 consider its next disorder value
                                correctValue += surfaces[b,s+k,w]
                            correctValue /= (n+1)

                            # fill the average value
                            for k in range(0, n+1):
                                surfacesCopy[b, s + k, w] = correctValue
                            s = s + n+1
                        else:
                            s += 1
            # after average the disorder near fovea, fill other values which may also be disorder
            surfacesCopy = torch.where(0 == surfacesCopy, surfaces, surfacesCopy)

            # final fully sort along column
            correctedSurfacesArray, _ = torch.sort(surfacesCopy, dim=-2)  # correct all surface constraint
            correctedSurfacesArray = correctedSurfacesArray.cpu().numpy()

            # load original images
            imagesList = glob.glob(volumesDir + f"/{patientVolumeName}/" + f"*_OCT[0-3][0-9].jpg")
            imagesList.sort()

            i = 0
            errorLocations = errorLocations.cpu().numpy()
            while i < currentErrorNum:
                s = errorLocations[i, 0]  # slice
                imageFile = imagesList[s]
                image = mpimg.imread(imageFile)
                if "5363_OD_25453" == patientName:
                    image = image[:, 103:615]
                else:
                    image = image[:, 128:640]  # for only middle image
                H, W = image.shape

                f = plt.figure(frameon=False)
                DPI = f.dpi
                f.set_size_inches(W * 3 / float(DPI), H / float(DPI))
                # a perfect solution for exact pixel size image.
                plt.margins(0)
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0,
                                    hspace=0)  # very important for erasing unnecessary margins.

                subplot1 = plt.subplot(1, 3, 1)
                subplot1.imshow(image, cmap='gray')
                errorLocationsTuple = np.nonzero(surfacesArray[s, 0:-1, :] > surfacesArray[s, 1:, :])  # return as tuple
                if len(errorLocationsTuple[0]) > 0:
                    subplot1.scatter(errorLocationsTuple[1],
                                     surfacesArray[s, errorLocationsTuple[0], errorLocationsTuple[1]], s=1, c='r',
                                     marker='o')
                subplot1.axis('off')

                subplot2 = plt.subplot(1, 3, 2)
                subplot2.imshow(image, cmap='gray')
                for surf in range(0, Num_Surfaces):
                    subplot2.plot(range(0, W), surfacesArray[s, surf, :], linewidth=0.4)
                subplot2.axis('off')

                subplot3 = plt.subplot(1, 3, 3)
                subplot3.imshow(image, cmap='gray')
                for surf in range(0, Num_Surfaces):
                    subplot3.plot(range(0, W), correctedSurfacesArray[s, surf, :], linewidth=0.4)
                subplot3.axis('off')

                titleName = patientName + f"_OCT{s + 1:02d}_ErrorMark_GT_Corrected"

                while i < currentErrorNum and errorLocations[i, 0] == s:
                    i += 1

                volumeDir = os.path.join(outputDir, patientVolumeName)
                if not os.path.exists(volumeDir):
                    os.makedirs(volumeDir)  # recursive dir creation

                plt.savefig(os.path.join(volumeDir, titleName + ".png"), dpi='figure', bbox_inches='tight',
                            pad_inches=0)
                plt.close()

    outputFile.write(
        f"\n\n=============== {errorPatients} patients with total {errorNum} locations have ground truth not conforming separation constraints ============\n")
    analyzeFile.write(
        f"\n\n=============== {errorPatients} patients with total {errorNum} locations have ground truth not conforming separation constraints ============\n")
    outputFile.close()
    analyzeFile.close()

    print("===========  End of program ===============\n\n")


if __name__ == "__main__":
    main()

