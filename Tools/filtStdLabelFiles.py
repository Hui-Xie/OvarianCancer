#  filt standard file with label
stdFileDir = "/home/hxie1/data/OvarianCancerCT/pixelSize223/nrrd"
labelFileDir = "/home/hxie1/data/OvarianCancerCT/rawNrrd/labels"
stdLabelFile = "/home/hxie1/data/OvarianCancerCT/pixelSize223withLabel/stdLabelFileList.json"

import sys
sys.path.append("..")
from FilesUtilities import *

stdFileList = getFilesList(stdFileDir,"_CT.nrrd")
stdFileSet = set([getStemName(x, "_CT.nrrd") for x in stdFileList])

labelFileList = getFilesList(labelFileDir,"_Seg.nrrd")
labelFileSet = set([getStemName(x, "_Seg.nrrd") for x in labelFileList])

stdLabelList = list(stdFileSet.intersection(labelFileSet))

# output standard files with label
import json
jsonData = json.dumps(stdLabelList)
f = open(stdLabelFile,"w")
f.write(jsonData)
f.close()







