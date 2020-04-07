import numpy as np

# %%
array1 = np.array([1, 2, 3])
print('array1 type : ', type(array1))
print('array1 array 형태 : ', array1.shape)

array2 = np.array([[1, 2, 3], [4, 5, 6]])
print('array2 type : ', type(array2))
print('array2 array 형태 : ', array2.shape)

array3 = np.array([[1, 2, 3]])
print('array3 type : ', type(array3))
print('array3 array 형태 : ', array3.shape)

# %%
print('array1 : %d차원, array2 : %d차원, array3 : %d차원' % (array1.ndim, array2.ndim, array3.ndim))

# %%
list1 = [1, 2, 3]
print(type(list1))

array1 = np.array(list1)
print(type(array1))
print(array1, array1.dtype)

# %%
sequence_array = np.arange(10)
# sequence_array = np.arange(stop=10, start=2)

print(sequence_array)
print(sequence_array.dtype, sequence_array.shape)

# %%
zero_array = np.zeros((3, 2), dtype='int32')
print(zero_array)
print(zero_array.dtype, zero_array.shape)

one_array = np.ones((3, 2))
print(one_array)
print(one_array.dtype, one_array.shape)

# %%
array1 = np.arange(10)
print('array1 :\n' + str(array1))

array2 = array1.reshape(2, 5)
print('array2 :\n' + str(array2))

array3 = array1.reshape(5, 2)
print('array3 : \n' + str(array3))

# %%
array1 = np.arange(10)
print(array1)

array2 = array1.reshape(-1, 5)
print('array2 shape :', array2.shape)

array3 = array1.reshape(5, -1)
print('array3 shape :', array3.shape)

# %%
array1 = np.arange(8)
array3d = array1.reshape((2, 2, 2))
print('array3d :\n' + str(array3d.tolist()))

array5 = array3d.reshape(-1, 1)
print('array5 :\n' + str(array5.tolist()))
print(array5.shape)
array6 = array1.reshape(-1, 1)
print('array6 :\n' + str(array6.tolist()))
print(array6.shape)
# %%
array1 = np.arange(start=1, stop=10)
print('array1 :', array1)

value = array1[2]
print(value)
print(type(value))

# %%
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3, 3)

print(array2d)

print('(row=0, col=0) index value :', array2d[0, 0])
print('(row=0, col=1) index value :', array2d[0, 1])
print('(row=1, col=0) index value :', array2d[1, 0])
print('(row=2, col=2) index value :', array2d[2, 2])

# %%
array1 = np.arange(start=1, stop=10)
array3 = array1[0:3]
print(array3)

# %%
array1 = np.arange(start=1, stop=10)
array4 = array1[:3]
array5 = array1[3:]
array6 = array1[:]
print(array4, array5, array6)

# %%
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3, 3)
print('array2d :\n' + str(array2d))

print('array2d[0:2, 0:2]\n', str(array2d[0:2, 0:2]))
print('array2d[1:3, 0:3]\n', str(array2d[1:3, 0:3]))
print('array2d[1:3, :]\n', str(array2d[1:3, :]))
print('array2d[:, :]\n', str(array2d[:, :]))
print('array2d[:2, 1:]\n', str(array2d[:2, 1:]))
print('array2d[:2, 0]\n', str(array2d[:2, 0]))

# %%
array1d = np.arange(start=1, stop=10)
array2d = array1d.reshape(3, 3)

array3 = array2d[[0, 1], 2]
print('array2d[[0, 1], 2]=>', array3.tolist())

array4 = array2d[[0, 1], 0:2]
print('array2d[[0, 1], 0:2]=>', array4.tolist())

array5 = array2d[[0, 1]]
print('array2d[[0, 1]]=>', array5.tolist())
# %%
array1d = np.arange(start=1, stop=10)
array3 = array1d[array1d > 5]
print(array3)
# %%
array2d = np.array([[8, 12], [7, 1]])
a = np.sort(array2d, axis=0)
print(a)

# %%
name_array = np.array(['John', 'Mike', 'Sarah', 'Kate', 'Samuel'])
score_array = np.array([78, 95, 84, 98, 88])

sort_indices_asc = np.argsort(score_array)
print('성적 오름차순 정렬 시 score_array의 인덱스 :', sort_indices_asc)
print('성적 오름차순으로 name_array의 이름 출력 :', name_array[sort_indices_asc])
# print('성적 오름차순으로 name_array의 이름 출력 :', name_array[np.argsort(score_array)])
