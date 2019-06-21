
import os
import SimpleITK as sitk
from DataMgr import DataMgr

suffix = "_CT.nrrd"
inputsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images"
outputImagesDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images_npy"
outputLabelsDir = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/labels_npy"
readmeFile = "/home/hxie1/data/OvarianCancerCT/Extract_ps2_2_5/images_npy/readme.txt"

originalCwd = os.getcwd()
os.chdir(inputsDir)
filesList = [os.path.abspath(x) for x in os.listdir(inputsDir) if suffix in x]
os.chdir(originalCwd)

Notes = "Notes: all files are corresponding with original nrrd files without any cropping.\n"

for file in filesList:
    patientID = DataMgr.getStemName(file, suffix)

    image = sitk.ReadImage(file)
    image3d = sitk.GetArrayFromImage(image)

    label = file.replace("_CT.nrrd", "_Seg.nrrd").replace("images/", "labels/")
    label3d = sitk.GetArrayFromImage(sitk.ReadImage(label))

    np.save(os.path.join(outputImagesDir, patientID + ".npy"), image3d)
    np.save(os.path.join(outputLabelsDir, patientID + ".npy"), label3d)

N = len(filesList)

with open(readmeFile,"w") as f:
    f.write(f"total {N} files in this directory\n")
    f.write(f"inputDir = {inputsDir}\n")
    f.write(f"inputImagesDir = {outputImagesDir}\n")
    f.write(f"inputLabelsDir = {outputLabelsDir}\n")
    f.write(Notes)

print("===End of convertNrrd to Npy=======")
