
# keep this:
'''

m = 2
nCount = 0
for a in range(-m, m+1):
    for b in range(-m, a+1):
        for c in range(-m, m+1):
            if (a==b and a>0) or (a==b and a==0 and c>=0):
                continue

            nCount +=1
            print(f"(a,b, c)={a,b,c}")

print(f"nCount = {nCount}")

'''


p=1
k=3
s=2

print(f" Din ,  Dout ,  Dout2 ")
for D in range(8,25):
    Dout = (D+2*p-k)//s + 1
    Dout2 = Dout*2
    print(f"{D}, {Dout}, {Dout2}")


import torch

X = torch.ones((3,4), requires_grad=True)
Z = torch.zeros((3,4))

for _ in range(0, 10):
    Y = X +2
    Z += Y*2

Z.backward(gradient=torch.ones(Z.shape))



print (X.grad)



print("==================")