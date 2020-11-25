# compare 2 ID files to make sure their data order consistent

idFile1 = "/home/hxie1/data/OCT_Duke/numpy_slices/log/SurfaceSubnet/expDuke_20201117A_SurfaceSubnet_NoReLU/testResult/validation/validation_surface_ID.txt"
idFile2 = "/home/hxie1/data/OCT_Duke/numpy_slices/log/RiftSubnet/expDuke_20200902A_RiftSubnet/testResult/validation/validation_Rift_ID.txt"

with open(idFile1, 'r') as f:
   IDList1 = f.readlines()
IDList1 = [item[0:-1] for item in IDList1]

with open(idFile2, 'r') as f:
   IDList2 = f.readlines()
IDList2 = [item[0:-1] for item in IDList2]

result = (IDList1 == IDList2)

if result:
    print(f"{idFile1} \n and {idFile2}\n are consistent!")
else:
    print(f"{idFile1} \n and {idFile2}\n are NOT consistent!")

