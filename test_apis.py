import torch
import pytorch_apis
#from pytorch_apis import vectorAdd
from pytorch_apis import Mul, Tpose


'''dim_0 = 3
a = torch.rand(dim_0)
b = torch.rand(dim_0)
device = torch.device("cuda")
a = a.to(device)
b = b.to(device)
c = vectorAdd(a, b, dim_0, 100000, device)
if torch.allclose(c, a + b): print("Computation on GPU is correct")
else: print("Computation on GPU is wrong")

print (c)'''
######

# Set the dimensions of the matrices
dim_m = 3  # Number of rows in matrix A
dim_n = 3  # Number of columns in matrix B
dim_k = 2  # Number of columns in matrix A / rows in matrix B

# Create random matrices A and B on the CPU
A = torch.rand(dim_m, dim_k, dtype=torch.float32)
print(A.size(0), A.size(1))
print("A 0,1",A.size(0), A.size(1)) # should be 3, 2
B = torch.rand(dim_k, dim_n, dtype=torch.float32)
print(A.size())

# Transfer matrices A and B to the GPU
device = torch.device("cuda")
A = A.to(device)
B = B.to(device)

# Call the Mul kernel to compute the product of A and B
C_gpu = Mul(A, B, dim_m, dim_n, device)  # int numARows, int numAColumns, int numBColumns
Ct_gpu = Tpose(A, dim_k, dim_m, device) #Arows, Acols
print("Ct Trans Size",Ct_gpu.size())

# Compute the expected result using PyTorch's matrix multiplication
print("###### Expected ######")
C_expected = torch.mm(A, B)
print(C_expected)
Ct_expected = torch.transpose(A, 0, 1)
print(Ct_expected)

# Transfer the result back to the CPU for comparison (if necessary)
C_gpu = C_gpu.to(torch.device("cpu"))
Ct_gpu = Ct_gpu.to(torch.device("cpu"))

# Compare the result computed by the GPU with the expected result
'''if torch.allclose(C_gpu, C_expected):
    print("Matrix multiplication on GPU is correct")
else:
    print("Matrix multiplication on GPU is incorrect")'''

# Print the result
print("###### Gained ######")
print(C_gpu)
print(Ct_gpu)
