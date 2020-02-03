import sys
sys.path.append("/home/hxie1/.local/lib/python3.6/site-packages")
import heyexReader

from scipy.io import loadmat

volFilePath  = "/home/hxie1/data/OCT_JHU/OCT_Manual_Delineations-2018_June_29/vol/hc01_spectralis_macula_v1_s1_R.vol"
matFilePath =  "/home/hxie1/data/OCT_JHU/OCT_Manual_Delineations-2018_June_29/delineation/hc01_spectralis_macula_v1_s1_R.mat"
#matFilePath = "/localscratch/Users/hxie1/temp/example.mat"
vol = heyexReader.volFile(volFilePath)


# vol.renderIRslo("slo.png", renderGrid = True)  # this line will generate SLO image.
# vol.renderOCTscans("oct", renderSeg = True)  # this line will generate a lot of BScans.

seg = loadmat(matFilePath)

segData =seg['control_pts']
a = segData[0,0]
b = segData[0,3]
print(vol.oct.shape)
print(vol.irslo.shape)