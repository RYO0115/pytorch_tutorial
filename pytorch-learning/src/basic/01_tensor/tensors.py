import torch
import numpy as np

# Directly from data
print("Directly from data")
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
print(x_data)

# From Numpy array
print("\nFrom Numpy array")
np_array = np.array([[1,2],[3,4]])
x_np = torch.from_numpy(np_array)
print(x_np)

# From other torch tensor
x_ones = torch.ones_like(x_data)
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(x_rand)

# With Random or Constant Value
shape = (2,3)
rand_tensor = torch.rand(shape)
print("Random Tensor :\n",rand_tensor)

ones_tensor = torch.ones(shape)
print("Ones Tensor:\n",ones_tensor)

zeros_tensor = torch.zeros(shape)
print("Zeros Tensor:\n",zeros_tensor)



# Tensor Attributes
print("Tensor Attributes")
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# Numpy like indexing and slicing
print("\nNumpy like indexing and slicing")
tensor = torch.ones(4,4)
print(tensor)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

tensor[:,1] = 0
print(tensor)

tensor[1,2] = 0
print(tensor)

# cat tensor
t1 = torch.cat([tensor,tensor,tensor], dim=1)
print("T1:\n",t1)

#Arithmetic operations
# Matrix multiplication
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)
print("y1:\n",y1)
print("y2:\n",y2)
print("y3:\n",y3)



# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)

print("z1:\n",z1)
print("z2:\n",z2)
print("z3:\n",z3)

# 

tensor_mul_result = torch.mul(tensor, tensor, out=z3)
print("Mul result:\n",tensor_mul_result)

print("\nOperations")
# torch.mul()
print(tensor.mul(tensor))
# @
print(tensor @ tensor)
# torch.matmul()
print(torch.matmul(tensor,tensor))

# torch.pow()
print(tensor.pow(2))

# Single-Element Tensor
agg = tensor.sum()
agg_item = agg.item()
print("Tensor Sum result:",agg_item, type(agg_item))

# In place operation
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

# Bridge with Numpy

# Tensor to Numpy (Shallow copy)
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# Tensor to Numpy (Clone)
print("\nClone Method\n")
t = torch.ones(5)

print(f"t: {t}")
n = t.clone().numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

np.add(t, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

# Numpy to Tensor
print("\nNumpy to Tensor")
np_array = np.array([1,2,3,4])
tensor = torch.from_numpy(np_array)
print(tensor)

np_array[0] = 0
print(np_array)
print(tensor)

# Numpy to Tensor (Clone)
print("\nNumpy to Tensor (Clone)")
np_array = np.array([1,2,3,4])
tensor = torch.from_numpy(np_array.copy())
print(tensor)

np_array[0] = 0
print(np_array)
print(tensor)
