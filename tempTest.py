
import SimpleITK as sitk

# file = "/home/hxie1/data/OvarianCancerCT/primaryROI/nrrd/04052781_pri.nrrd"  # error axis file.
file = "/home/hxie1/data/OvarianCancerCT/primaryROI/nrrd/05055556_pri.nrrd"

image = sitk.ReadImage(file)
image3d = sitk.GetArrayFromImage(image)

print(f"Shape: {image3d.shape}")


