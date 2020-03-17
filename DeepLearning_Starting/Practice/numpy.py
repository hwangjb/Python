import numpy as np
# %%
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


# %%
a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[-1, -2], [-3, -4], [-5, -6]])

c = np.dot(a, b) # 행렬의 곱

print("A.shape ==", a.shape, "B.shape ==", b.shape)
print("C.shape ==", c.shape)
print(c)

# %%
# numpy broadcast : 크기가 다른 두 행렬간에도 사칙연산을 할수 있음(행렬곱에서는 적용 안됨!!!!)

a = np.array([[1, 2],[3, 4]])
b = 5
c = np.array([4, 5])

print(a+b)
print(a+c)

# %%

a = np.array([[1, 2], [3, 4], [5, 6]])

b = a.T
print(a.shape, b.shape)
print(a)
print(b)

# %%
A = np.array([[10, 20, 30, 40],[50, 60, 70, 80]])

it = np.nditer(A, flags=['multi_index'], op_flags=['readwrite'])

while not it.finished:
    idx = it.multi_index
    print("current value =>", A[it.multi_index])
    it.iternext()


# %%
# concatenate 함수 : 서로 다른 행렬을 합쳐줌
a = np.array([[1, 2, 3], [4, 5, 6]])

row_add = np.array([7, 8, 9]).reshape(1,3)
b = np.concatenate((a, row_add), axis=0)
print(b)

colum_add = np.array([10, 11]).reshape(2, 1)
c = np.concatenate((a, colum_add), axis=1)
print(c)

# %%
loaded_data = np.loadtxt('./dataset/data-01.csv', delimiter=',', dtype=np.float32)
print(loaded_data)


x_data = loaded_data[:, 0:-1]
print(x_data)

y_data = loaded_data[:, [-1]]
print(y_data)

# %%
random_number1 = np.random.rand(3)
random_number2 = np.random.rand(1, 3)
random_number3 = np.random.rand(3, 1)

print(random_number1, random_number1.shape)
print(random_number2, random_number2.shape)
print(random_number3, random_number3.shape)

# %%
x = np.array([[2, 4, 6], [1, 2, 3], [0, 5,8]])

print(np.max(x, axis=0))
print(np.min(x, axis=0))
print(np.max(x, axis=1))
print(np.min(x, axis=1))
print(np.argmax(x, axis=0))
print(np.argmin(x, axis=0))
print(np.argmax(x, axis=1))
print(np.argmin (x, axis=1))

a = np.ones([3,3])
b = np.zeros([3,2])
print(a.shape, a)
print(b.shape, b)


