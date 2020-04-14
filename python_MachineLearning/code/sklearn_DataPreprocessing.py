# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

# %%
items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']

encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
print('인코딩 변환값 :', labels)

# %%
print('encoding class :', encoder.classes_)
print('decoding values :', encoder.inverse_transform([0, 1, 4, 5, 3, 3, 2, 2]))

# %%
items = ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']
encoder = LabelEncoder()
encoder.fit(items)
labels = encoder.transform(items)
labels = labels.reshape(-1, 1)

oh_encoder = OneHotEncoder()
oh_encoder.fit(labels)
oh_labels = oh_encoder.transform(labels)
print('one-hot encoder data')
print(oh_labels.toarray())
print('one hot encoder shape')
print(oh_labels.shape)

# %%
df = pd.DataFrame({'items': ['TV', '냉장고', '전자레인지', '컴퓨터', '선풍기', '선풍기', '믹서', '믹서']})
print(pd.get_dummies(df))

# %%
iris = load_iris()
iris_data = iris.data
iris_df = pd.DataFrame(data= iris_data, columns=iris.feature_names)

print('feature mean')
print(iris_df.mean())
print('feature var')
print(iris_df.var())

# %%
scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data= iris_scaled, columns=iris.feature_names)
print('feature mean')
print(iris_df_scaled.mean())
print('feature var')
print(iris_df_scaled.var())

# %%
scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data=iris_scaled, columns=iris.feature_names)
print('feature Min')
print(iris_df_scaled.min())
print('feature Max')
print(iris_df_scaled.max())

# %%
train_array = np.arange(0, 11).reshape(-1, 1)
test_array = np.arange(0, 6).reshape(-1, 1)

scaler = MinMaxScaler()
scaler.fit(train_array)
train_scaled = scaler.transform(train_array)
print('원본 train_array data :', train_array.reshape(-1))
print('Scale된 train_array data :', train_scaled.reshape(-1))

# %%
scaler.fit(test_array)
test_scaled = scaler.transform(test_array)
print('원본 test_array data :', test_array.reshape(-1))
print('Scale된 test_array data :', test_scaled.reshape(-1))

# %%
scaler.fit(train_array)
test_scaled = scaler.transform(test_array)
train_scaled = scaler.transform(train_array)
print('원본 train_array data :', train_array.reshape(-1))
print('Scale된 train_array data :', train_scaled.reshape(-1))
print('\n\n원본 test_array data :', test_array.reshape(-1))
print('Scale된 test_array data :', test_scaled.reshape(-1))
