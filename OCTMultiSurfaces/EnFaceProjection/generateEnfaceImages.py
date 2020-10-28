# generate en-face images from segmentation result and OCT volume

imageIDPath = "/home/hxie1/data/BES_3K/GTs/allID_delNonExist_delErrWID_excludeMGM.csv"

OCTVolumeDir = ""
segXmlDir =""
outputDir =""


# read volumeID into list
with open(imageIDPath, 'r') as idFile:
    IDList = idFile.readlines()
IDList = [item[0:-1] for item in IDList]  # erase '\n'

# read xml segmentation into array

# read volume

# define output emtpy array

# fill the output array

# output file

