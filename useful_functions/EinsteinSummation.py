import numpy as np

# Prepare toy tensors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# --- 1. Vector Dot Product (ベクトルの内積) ---
# Summing over 'i' because it's not present in the output
dot = np.einsum("i,i->", a, b)             # 1*4 + 2*5 + 3*6 = 32
print("Dot Product:", dot)                 # np.dot(a, b)

# --- 2. Outer Product (ベクトルの外積 / 直積) ---
# Keeping both 'i' and 'j' in the output, so no summation occurs
outer = np.einsum("i,j->ij", a, b)
print("Outer Product:\n", outer)           # np.outer(a, b)

# --- 3. Matrix Transpose (行列の転置) ---
# Rearranging the index order
transpose = np.einsum("ij->ji", A)
print("Transpose:\n", transpose)           # A.T

# --- 4. Matrix Multiplication (行列積) ---
# 'j' is in both inputs but missing from output, so we sum over it: Out[i,k] = sum_j (A[i,j] * B[j,k])
matmul = np.einsum("ij,jk->ik", A, B)
print("Matrix Mult:\n", matmul)            # A @ B

# --- 5. Trace (対角成分の和) ---
# Matching indices on the same matrix automatically extracts the diagonal and sums
trace = np.einsum("ii->", A)               # 1 + 4 = 5
print("Trace:", trace)                     # np.trace(A)
