import pydicom
from pydicom.filereader import read_dicomdir

visitDir = "/home/hxie1/temp/testDicom/1.2.276.0.75.2.2.42.896740156037.20170213165203155.121661155.1.dcm"

patiendDicom = read_dicomdir(visitDir)

print("===========================")