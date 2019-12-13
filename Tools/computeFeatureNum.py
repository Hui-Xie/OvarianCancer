
import operator as op
from functools import reduce
import math

p1 = 0.57
p2 = 1-p1
N = 192   # total feature number

def nCr(n, r):   # combination number of n Choose r
    if r==0 or r==n:
        return 1
    else:
        r = min(r, n-r)
        numer = reduce(op.mul, range(n, n-r, -1), 1)
        denom = reduce(op.mul, range(1, r+1), 1)
        return numer / denom

a = 0.0   # accumulate correct probability
for i in range(N//2+1, N+1, 1):
    a += nCr(N, i)*math.pow(p1,i)*math.pow(p2,N-i)

print(f"p1={p1}")
print(f"N= {N}")
print(f"accumulative majority correct probability: {a}")
