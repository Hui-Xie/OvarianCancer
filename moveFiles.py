# move file from one directory to another
import os
import subprocess

srcDir = '/home/hxie1/data/OvarianCancerCT/Extract/images'
dstDir = '/home/hxie1/data/OvarianCancerCT/Extract/nonStdImages'

# non-standard files which are not suitable for trainding and test
fileList= ['03020770','03389601', '03540492','03608463','03741527','03985076','05056196','05430021','05431967',
           '05612548', '05612585','05739688','58072163','70504698','72209667','73221833','79292928','83168286',
           '88032071','99170611', '99669937',                     #  above is from the 1st batch data
           '01651501','02974861','05252246','06083031','06142141','06934663','07244681','07352839','07651600',
           '08388679','10145066','11155987','11451845','14028617','14061070','69161236','76408944','77159277',
           '89300137','92222822'                     # above is from the 2nd batch data
           ]


n = 0
for file in fileList:
    filename = file + "_CT.nrrd"
    srcFile = os.path.join(srcDir, filename)
    dstFile = os.path.join(dstDir, filename)
    if os.path.isfile(srcFile):
        subprocess.call(["mv", srcFile, dstFile])
        n +=1
    else:
        print(f"{srcFile} does not exist.")

print(f"total {n} files have been moved from {srcDir} to {dstDir}.")


