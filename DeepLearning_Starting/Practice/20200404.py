# %%
import pandas as pd
train = pd.read_csv('C:/Users/user/Desktop/(1.학습)KISA-challenge2019-Network_trainset/(공격샘플)03.Brute_Force_attack_sample.csv')
print(train.shape)
print(train.columns)
print(train['tcp.srcport'])

