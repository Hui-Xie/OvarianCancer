import sys
sys.path.append("/home/hxie1/.local/lib/python3.6/site-packages")
import heyexReader

volFilePath  = "/home/hxie1/data/OCT_JHU/OCT_Manual_Delineations-2018_June_29/vol/hc01_spectralis_macula_v1_s1_R.vol"

vol = heyexReader.volFile(volFilePath)

vol.renderIRslo("slo.png", renderGrid = True)
vol.renderOCTscans("oct", renderSeg = True)

print(vol.oct.shape)
print(vol.irslo.shape)