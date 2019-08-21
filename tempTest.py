
import torch

A = torch.tensor([[1,2,3],[3,4,5]],requires_grad= True, dtype=torch.float)

b = torch.tensor([0.5,2,1])

print("Before backward: A.grad", A.grad)

B = A.clone()
C = B.clone()

y = torch.mean(C*b)
y.backward()

print("After backward, A.grad", A.grad)


#y = torch.mean(A*b)
#y.backward()

# print(A.grad)
# tensor([[0.0833, 0.3333, 0.1667],
#         [0.0833, 0.3333, 0.1667]])



