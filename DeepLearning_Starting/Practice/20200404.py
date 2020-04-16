# %%
import pandas as pd
train = pd.read_csv('C:/Users/user/Desktop/Dataset/(1.학습)KISA-challenge2019-Network_trainset/(분할파일)network_train_set1_분할/network_train_set1_00000.csv')
# %%
print(train.shape)
print(train.isnull().sum())
