import numpy as np

A = np.array([ [1, 0], [0, 1] ])
B = np.array([ [1, 1], [1, 1] ])

print(A)
print(B)
print(A+B)

# %%
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

print(A)
print(B)

print(A.shape, B.shape) #원소 개수
print(A.ndim, B.ndim) #차원

print(A+B)
print(A-B)
print(A*B)
print(A/B)

# %%

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[-1, -2, -3], [-4, -5, -6]])

print(a.shape, b.shape)
print(a.ndim, b.ndim)


c = np.array([1, 2, 3])
print(c.shape)
print(c)

c = c.reshape(1,3)
print(c.shape)
print(c)
