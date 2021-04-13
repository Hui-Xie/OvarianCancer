# Test large matrix inverse

# for Duke data: B= 8, S=3, W=361. Its maximum matrix size: BxSWxSW: 8x1083x1083
# for Tongren data: B=4, S=11, W=512. Its maximum matrix size: BxSWxSW: 4x5632x5632

import torch
import time

B=4  # batch size
S=11  # number of surfaces
W=512  # number of image column
device = torch.device("cuda:3")

startTime = time.time()
M = torch.rand((B,S*W,S*W), device=device,requires_grad=True)
identityM = torch.eye(S*W,S*W, device=device)*0.01
identityM = identityM.unsqueeze(dim=0)
identityM = identityM.expand_as(M)
M = M+identityM
MInv = torch.inverse(M)
memorysummary = torch.cuda.max_memory_allocated(device=device)
runTime = time.time()-startTime
print(f"Matrix shape: {M.shape}")
print(f"===Matrix Inverse running time: {runTime:.2f} seconds.")  # about 3.9 seconds.
print(f"memory usage:  {memorysummary:,} byte\n")  #1.6GB
verify =  torch.bmm(M, MInv)

print(f"verify[0, 1000,1000] (should be 1) ={verify[0,1000,1000]} ")
print(f"verify[0, 1000,100] (should be 0)  ={verify[0, 1000,100]} ")



'''
# general matrix inverse result:
Matrix shape: torch.Size([10240, 10240])
===Matrix Inverse running time: 3.9061315059661865 seconds.
memory usage:  1680343040 byte

matrix memory: 10240*10240*4 = 419430400 byte = 419MB

==============
Matrix shape: torch.Size([12288, 12288])
===Matrix Inverse running time: 4.199603319168091 seconds.
memory usage:  2419064832 byte

verify[1000,1000] =0.9997560977935791 
verify[1000,100] =5.3988151194062084e-05 


for  Duke data: B= 8, S=3, W=361. Its maximum matrix size: BxSWxSW: 8x1083x1083
Matrix shape: torch.Size([8, 1083, 1083])
===Matrix Inverse running time: 2.44 seconds.
memory usage:  155,686,912 byte

verify[0, 1000,1000] (should be 1) =0.9999899864196777 
verify[0, 1000,100] (should be 0)  =-1.238885033671977e-05 




'''